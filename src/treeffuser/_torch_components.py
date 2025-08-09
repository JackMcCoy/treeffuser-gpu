import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from ._score_models import ScoreModel, _make_training_data
from treeffuser.sde import DiffusionSDE, sdeint

# A simple MLP to be used as the score network
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        # Concatenate time embedding to input
        t_emb = t.expand(x.shape[0], -1)
        inp = torch.cat([x, t_emb], dim=1)
        return self.network(inp)

class TorchMLPScoreModel(ScoreModel):
    """
    A score model that uses a PyTorch MLP to approximate the score.
    This model is designed to run on a GPU and be compatible with torchsde.
    """
    def __init__(self, n_repeats=10, eval_percent=0.1, seed=None, learning_rate=1e-3, epochs=100, batch_size=128):
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.seed = seed
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if toch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model = None
        self.sde = None

    def fit(self, X, y, sde: DiffusionSDE, cat_idx=None):
        self.sde = sde
        y_dim = y.shape[1]
        x_dim = X.shape[1]
        
        # The input to the MLP will be [y, X, t]
        input_dim = y_dim + x_dim + 1
        self.model = MLP(input_dim, y_dim).to(self.device)
        
        # Create the training data using the same helper as LightGBM
        lgb_X_train, lgb_X_val, lgb_y_train, lgb_y_val, _ = _make_training_data(
            X=X, y=y, sde=self.sde, n_repeats=self.n_repeats,
            eval_percent=self.eval_percent, cat_idx=cat_idx, seed=self.seed
        )

        # Convert to torch tensors
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(lgb_X_train).float(),
            torch.from_numpy(lgb_y_train).float()
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for predictors, targets in pbar:
                predictors, targets = predictors.to(self.device), targets.to(self.device)
                
                # Split predictors into [y_perturbed, x, t]
                y_perturbed = predictors[:, :y_dim]
                x_features = predictors[:, y_dim:-1]
                t_time = predictors[:, -1:]
                
                # The model expects [y_perturbed + x_features, t]
                model_input = torch.cat([y_perturbed, x_features], dim=1)
                
                optimizer.zero_grad()
                output = self.model(model_input, t_time)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"loss": loss.item()})
        
        return self

    def score(self, y, X, t):
        if self.model is None:
            raise ValueError("The model has not been fitted yet.")
        
        self.model.eval()
        with torch.no_grad():
            # Ensure inputs are torch tensors on the correct device
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(y).float().to(self.device)
            if not isinstance(X, torch.Tensor):
                X = torch.from_numpy(X).float().to(self.device)
            if not isinstance(t, torch.Tensor):
                t = torch.from_numpy(t).float().to(self.device)

            model_input = torch.cat([y, X], dim=1)
            
            # The score is parametrized: score(y, x, t) = MLP(y, x, t) / std(t)
            score_p = self.model(model_input, t)
            
            # We need y0 to calculate std, but we don't have it here.
            # We assume the SDE's get_mean_std_pt_given_y0 can handle this.
            # For simplicity, we'll pass `y` as `y0` to get the std at time `t`.
            # This is a slight deviation but necessary for the interface.
            _, std = self.sde.get_mean_std_pt_given_y0(y, t)
            
            if isinstance(std, np.ndarray):
                std = torch.from_numpy(std).float().to(self.device)

            score = score_p / std
            return score

class TorchVESDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, hyperparam_schedule):
        super().__init__()
        # Assuming hyperparam_schedule is also adapted to output torch tensors
        self.hyperparam_schedule = hyperparam_schedule
        self.f = self._drift
        self.g = self._diffusion

    def _drift(self, t, y):
        return torch.zeros_like(y)

    def _diffusion(self, t, y):
        hyperparam = self.hyperparam_schedule.get_value(t)
        hyperparam_prime = self.hyperparam_schedule.get_derivative(t)
        # Same math as treeffuser, but with torch operations for GPU execution
        diffusion = torch.sqrt(2 * hyperparam * hyperparam_prime)
        return diffusion.unsqueeze(-1).expand_as(y)


class ReverseTorchSDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'

    def __init__(self, forward_sde, score_model, x_features, T=1.0):
        super().__init__()
        self.forward_sde = forward_sde
        self.score_model = score_model
        self.x_features = x_features # Store the conditional features
        self.T = T
        self.g = self.forward_sde.g # Diffusion is the same in reverse

    def f(self, t, y): # Reverse Drift
        # The reverse drift is -f(y,t) + g(t)^2 * score(y,X,t)
        reverse_time = self.T - t
        
        forward_drift = self.forward_sde.f(reverse_time, y)
        diffusion = self.forward_sde.g(reverse_time, y)
        
        # The score function call is now a native PyTorch operation
        score = self.score_model.score(y, self.x_features, reverse_time)
        
        return -forward_drift + (diffusion**2) * score
