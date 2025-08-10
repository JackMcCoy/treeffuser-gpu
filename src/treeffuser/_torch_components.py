import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import copy
import math

from ._base_score_model import ScoreModel
from ._training_utils import _make_training_data
from treeffuser.sde import DiffusionSDE
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_torch_device():
    """
    Determines the most appropriate torch device to use.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class ResidualBlock(nn.Module):
    """
    A simple residual block with two linear layers.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        
        self.activation = nn.SiLU() # SiLU (Swish) is a common and effective choice
        
        if input_dim != output_dim:
            self.residual_connection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        h = self.activation(self.linear1(x))
        h = self.linear2(h) 
        
        return h + self.residual_connection(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# A simple MLP to be used as the score network
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=4, time_emb_dim=32):
        super().__init__()
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.Mish(),
        )
        self.initial_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)        
        self.blocks = nn.ModuleList()
        current_dim = hidden_dim
        for dim in range(n_layers):
            self.blocks.append(ResidualBlock(
                input_dim=hidden_dim, 
                output_dim=hidden_dim
            ))
        self.final_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.squeeze(-1))
        h = self.initial_proj(torch.cat([x, t_emb], dim=1))
        
        for block in self.blocks:
            x_with_time = h+t_emb
            h = block(x_with_time)
            
        return self.final_proj(h)

class TorchMLPScoreModel(ScoreModel, nn.Module):
    """
    A score model that uses a PyTorch MLP to approximate the score.
    This model is designed to run on a GPU and be compatible with torchsde.
    """
    def __init__(self, n_repeats=10, eval_percent=0.1, seed=None, learning_rate=1e-3, epochs=100, batch_size=128, early_stopping_rounds=10):
        super().__init__()
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.seed = seed
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.device = get_torch_device()

        self.model = None
        self.sde = None

    def fit(self, X, y, sde: DiffusionSDE, cat_idx=None):
        self.sde = sde
        y_dim = y.shape[1]
        x_dim = X.shape[1]
        
        input_dim = y_dim + x_dim
        self.model = MLP(input_dim, y_dim).to(self.device)
        
        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val, _ = _make_training_data(
            X=X, y=y, sde=self.sde, n_repeats=self.n_repeats,
            eval_percent=self.eval_percent, cat_idx=cat_idx, seed=self.seed
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(lgb_X_train).float(),
            torch.from_numpy(lgb_y_train).float()
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if lgb_X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(lgb_X_val).float(),
                torch.from_numpy(lgb_y_val).float()
            )
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=self.lr,
                                                    total_steps=self.epochs * (len(X) // self.batch_size),
                                                    pct_start=0.03, # Use 10% of steps for warmup
                                                    anneal_strategy='cos')

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        early_stopped = False

        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for predictors, targets in pbar:
                predictors, targets = predictors.to(self.device), targets.to(self.device)
                
                y_perturbed = predictors[:, :y_dim]
                x_features = predictors[:, y_dim:-1]
                t_time = predictors[:, -1:]
                
                model_input = torch.cat([y_perturbed, x_features], dim=1)
                
                optimizer.zero_grad()
                output = self.model(model_input, t_time)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})

            if val_loader and self.early_stopping_rounds:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for predictors_val, targets_val in val_loader:
                        predictors_val, targets_val = predictors_val.to(self.device), targets_val.to(self.device)
                        y_perturbed_val = predictors_val[:, :y_dim]
                        x_features_val = predictors_val[:, y_dim:-1]
                        t_time_val = predictors_val[:, -1:]
                        model_input_val = torch.cat([y_perturbed_val, x_features_val], dim=1)
                        output_val = self.model(model_input_val, t_time_val)
                        val_loss += criterion(output_val, targets_val).item()
                
                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.early_stopping_rounds:
                    print(loss.item())
                    print(f"Early stopping at epoch {epoch + 1}")
                    early_stopped = True
                    break

            scheduler.step()
        
        if early_stopped and best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self

    def score(self, y, X, t):
        # y, X, and t are expected to be torch.Tensors on the correct device
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")
        
        self.model.eval()
        with torch.no_grad():
            t_clamped = torch.clamp(t, min=1e-5)
            model_input = torch.cat([y, X], dim=1)
            score_p = self.model(model_input, t_clamped)
            
            alpha_t = self.sde.hyperparam_schedule.min_value * \
                      (self.sde.hyperparam_schedule.max_value / self.sde.hyperparam_schedule.min_value) ** t_clamped
            alpha_0 = torch.full_like(alpha_t, self.sde.hyperparam_schedule.min_value)

            std = (alpha_t**2 - alpha_0**2).sqrt()

            if std.shape != score_p.shape:
                 std = std.expand_as(score_p)

            # Safeguard against division by zero
            score = score_p / torch.clamp(std, min=1e-5)
            
        return score.detach().cpu().numpy()

class TorchVESDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, hyperparam_schedule):
        super().__init__()
        self.hyperparam_schedule = hyperparam_schedule
        self.f = self._drift
        self.g = self._diffusion

    def _drift(self, t, y):
        return torch.zeros_like(y)

    def _diffusion(self, t, y):
        hyperparam = self.hyperparam_schedule.get_value(t)
        hyperparam_prime = self.hyperparam_schedule.get_derivative(t)
        diffusion = torch.sqrt(2 * hyperparam * hyperparam_prime)
        # Reshape to (batch, state, noise) for torchsde with scalar noise
        return diffusion.view(-1, 1, 1).expand(y.size(0), y.size(1), 1)


class ReverseTorchSDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, forward_sde, score_model, x_features, T=1.0):
        super().__init__()
        self.forward_sde = forward_sde
        self.score_model = score_model
        self.x_features = x_features 
        self.T = T
        #self.g = self.forward_sde.g

    def f(self, t, y): # Reverse Drift
        reverse_time = self.T - t
        forward_drift = self.forward_sde.f(reverse_time, y)
        diffusion = self.forward_sde.g(reverse_time, y)
        
        # Reshape the scalar t from torchsde to a 2D tensor for the score model
        t_for_score = reverse_time.expand(y.size(0), 1)
        score = self.score_model.score(y, self.x_features, t_for_score)
        
        # The diffusion tensor is (batch, state, noise). We need to match it with the score's shape (batch, state).
        # For scalar noise, the noise dimension is 1, so we can squeeze it.
        diffusion_term = (diffusion**2).squeeze(-1)
        
        return -forward_drift + diffusion_term * score

    def g(self, t, y):
        reverse_time = self.T - t
        return self.forward_sde.g(reverse_time, y)
