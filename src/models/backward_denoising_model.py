# Common
import numpy as np
from matplotlib import rc
rc('axes', linewidth=1)

from functools import partial

# Tensorcircuit
import tensorcircuit as tc
from tensorcircuit.densitymatrix import DMCircuit

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

# Pytorch
import torch
import torch.nn as nn

from .. import utils

def backward_circuit(density_matrix, params, n, n_ancilla, L):
    total = n + n_ancilla
    qc = DMCircuit(total, dminputs = density_matrix)
    for l in range(L):
        for i in range(total):
            qc.rx(i, theta=params[2*l*total+2*i])
            qc.ry(i, theta=params[2*l*total+2*i+1])
            
        for i in range(total//2):
            qc.cz(2*i, 2*i+1)
        for i in range((total-1)//2):
            qc.cz(2*i+1, 2*i+2)

    return qc.state()


class BackwardDenoisingModel(nn.Module):
    
    def __init__(self, n, n_ancilla, T, L, use_zero_ancillas=False):
        super().__init__()
        
        self.n = n
        self.n_ancilla = n_ancilla
        self.n_total = n + n_ancilla
        self.T = T
        self.L = L
        self.target_states = None
        self.use_zero_ancillas = use_zero_ancillas
        # Embed the circuit to a vectorized pytorch neural network layer
        self.backward_circuit_vmap = K.vmap(partial(backward_circuit, n = self.n, n_ancilla = self.n_ancilla, L = L), vectorized_argnums=0)

    def set_target_states(self, target_states):
        self.target_states = torch.from_numpy(target_states).cfloat()

    def forward(self, x, params, haars=None):
        out = torch.zeros((len(x), 2**self.n, 2**self.n)).cfloat()

        # x = (n_train, 2**(n + n_ancilla), 2**(n + n_ancilla)) with first n_ancilla qubits as ancillas
        if self.use_zero_ancillas:
            ancilla = torch.zeros((2**(self.n_ancilla), 2**(self.n_ancilla))).cfloat()
            ancilla[0, 0] = 1.
            out1 = torch.kron(ancilla, x[:, :2**self.n, :2**self.n])
        else:
            ancilla = torch.zeros((2**(self.n_ancilla - 1), 2**(self.n_ancilla - 1))).cfloat()
            ancilla[0,0] = 1.
            out1 = torch.einsum('nui,vj,nwk->nuvwijk', haars, ancilla, x[:, :2**self.n, :2**self.n]).reshape(x.shape[0], 2 ** (self.n_total), 2 ** (self.n_total))
        
        if torch.any(torch.isnan(out1)):
            print("Init failed before circuit. use_zero_ancillas: {self.use_zero_ancillas}; x: {x[:,:2**self.n, :2**self.n]}; out1: {out1}; params: {params}")
        
        out2 = self.backward_circuit_vmap(out1, params)
        
        if torch.any(torch.isnan(out2)):
            print("Init failed after circuit. x: {x[:,:2**self.n, :2**self.n]}; out1: {out1}; out2: {out2}; params: {params}")

        # Perform projective measurement
        out = self._random_measure(out2)

        if torch.any(torch.isnan(out)):
            print(f"Init failed after projective measurement. x: {x}; out: {out}; params: {params}")

        return out # (n_train, 2**n, 2**n)
    
    def _random_measure(self, density_matrices):
        # Calculate measurement probabilities
        probs = torch.abs(torch.diagonal(density_matrices, dim1=-2, dim2=-1)).reshape(density_matrices.shape[0], -1, 2**self.n).sum(dim=-1)
        m_res = torch.multinomial(probs, num_samples=1).squeeze()

        indices = ((2**self.n) * m_res.view(-1, 1) + torch.arange(2**(self.n)))
        post_state = torch.zeros(density_matrices.shape[0], 2**self.n, 2**self.n).cfloat()
        for i in range(indices.shape[0]):
            post_state[i] = density_matrices[i, indices[i, 0]:(indices[i, -1]+1), indices[i, 0]:(indices[i, -1]+1)]
        for i in range(2**self.n):
            post_state[:,i,i]=torch.real(post_state[:,i,i].clone())
        norms = torch.sum(torch.real(torch.diagonal(post_state, dim1 = -2, dim2 = -1)), axis=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        state = 1. * post_state / norms
        return state

    def prepare_backward_states(self, t, init_T_states, params):
        """
        Return quantum states in backward denoising step t (psi_t+1) with initial states 
        [Params] t: step, init_T_states: input states (numpy arrays) at step T
        If t == T, empty circuit.
        """
        n_train = len(init_T_states)
        init_tplus1s = torch.zeros((n_train, 2**self.n, 2**self.n)).cfloat()
        
        with torch.no_grad():
            init_tplus1s = init_T_states
            for k in range(self.T - t):
                if not self.use_zero_ancillas:
                    haars = torch.from_numpy(utils.get_haar_density_matrices(1, n_train)).cfloat()
                else:
                    haars = None
                init_tplus1s = self.forward(init_tplus1s, params[k], haars)

        return init_tplus1s
