# Common
import numpy as np

# Pytorch
import torch
import torch.nn as nn


class ForwardDiffusionModel(nn.Module):
    """
    The forward quantum circuit model.
    Apply depolarizing channels to make the state totally mixed.
    Args:
    n: number of qubits
    T: number of diffusion steps
    n_data: number of data
    """
    def __init__(self, n, T, n_data, beta_start = None, beta_end = None, beta_schedule = 'linear'):
        super().__init__()
        self.n = n
        self.T = T
        self.n_data = n_data
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

    def _depolarizing_channel(self, t, init_dm, ps, gs=None):
        '''
        Return the state for diffusion step t
        Args:
        t: diffusion step
        init_dm: initial quantum state (density matrix)
        ps: the single-qubit depolarizing parameters in diffusion circuit
        '''
        n_dim = 2 ** (self.n)
        result = np.copy(init_dm)

        for step in range(t):
            for i in range(self.n):
                p = 1. * ps[self.n*step + i].item()
                result = (1-p) * result + p / n_dim * np.eye(n_dim)

        return result

    def _get_betas(self, t):
        if self.beta_schedule == 'linear':
            if self.beta_start is not None:
                betas = torch.from_numpy(np.linspace(self.beta_start, self.beta_end, self.T)[:t])
            else:
                betas = torch.ones(t)
        elif self.beta_schedule == 'cosine':
            betas = self._get_cosine_schedule(self.T, s=0.001)[:t]
        elif self.beta_schedule == 'sq_cosine':
            betas = self._get_cosine_schedule(self.T, s=0.001)[:t]
            betas = betas ** 2
        else:
            print("[ERROR] Wrong beta scheduling; beta_schedule should be 'linear', 'cosine', or 'sq_cosine'.")
            betas = torch.ones(t)
        betas = betas.unsqueeze(1).repeat((1, self.n)).reshape(-1)
        return betas

    def _get_cosine_schedule(self, num_timesteps, s=0.0001):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor([0]))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 1.)
        return betas
    
    def get_forward_density_matrices(self, t, init_dms, seed = None):
        '''
        return the quantum data set after step t of the forward diffusion
        Args:
        t: diffusion step
        init_dms: the input quantum data set (in density matrix format)
        seed: random seed
        '''
        # set single-qubit depolarizing noises
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        
        betas = self._get_betas(t)
        p_noises = torch.ones(self.n_data, self.n * t) * 1.0
        p_noises *= betas
        
        states = torch.zeros((self.n_data, 2**self.n, 2**self.n)).cfloat()
        for i in range(self.n_data):
            states[i] = torch.from_numpy(self._depolarizing_channel(t, init_dms[i], p_noises[i]))

        return states
