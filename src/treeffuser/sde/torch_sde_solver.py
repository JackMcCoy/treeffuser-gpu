import torch
import torchsde
import numpy as np

from .base_solver import BaseSDESolver, _register_solver

@_register_solver(name="torchsde")
class TorchSolver(BaseSDESolver):
    """
    An SDE solver that uses the torchsde library to perform integration on a GPU.
    This solver expects the SDE and score_fn to be torch.nn.Modules.
    """
    def __init__(self, sde, n_steps, seed=None):
        super().__init__(sde, n_steps, seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if seed is not None:
            torch.manual_seed(seed)

    def integrate(self, y0, t0, t1):
        # This method assumes self.sde is already a torch.nn.Module (the ReverseTorchSDE)
        if not isinstance(self.sde, torch.nn.Module):
            raise TypeError("The 'torchsde' method requires a PyTorch-native SDE module.")
            
        y0_torch = torch.from_numpy(y0).float().to(self.device)
        
        # In the reverse process, t0=T and t1=0.
        # torchsde integrates forward from 0 to T.
        ts = torch.tensor([0., t0], device=self.device)
        
        # Perform the entire integration in one GPU call
        solution = torchsde.sdeint(self.sde, y0_torch, ts, dt=1./self.n_steps, method='euler')
        
        # Return the final state, converted back to a NumPy array
        return solution[-1].cpu().numpy()

    def step(self, y0, t0, t1):
        raise NotImplementedError("TorchSolver overrides integrate() and does not use step().")
