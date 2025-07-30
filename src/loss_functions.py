# Common
import numpy as np
from matplotlib import rc
rc('axes', linewidth=1)

from opt_einsum import contract

# Tensorcircuit
import tensorcircuit as tc

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

# Pytorch
import torch
import torch.nn as nn

import ot

eps = torch.tensor([1e-16])

def get_loss_fn(loss_type='mmd', fidelity_type='super'):
    if loss_type == 'mmd':
        loss_fn = MMDLoss(fidelity_type)
    elif loss_type == 'wasserstein':
        loss_fn = WassersteinLoss(fidelity_type)
    else:
        loss_fn = MMDLoss(fidelity_type)

    return loss_fn

class MMDLoss(nn.Module):
    def __init__(self, fidelity_type='super'):
        super(MMDLoss, self).__init__()
        self.fidelity_type = fidelity_type # normal, super
        print(f"[MMDLoss] Use {fidelity_type} fidelity in cost function")

    def sqrt(self, mat: torch.Tensor) -> torch.Tensor:
        """Square root of a positive semidefinite matrix mat."""
        eigs, vecs = torch.linalg.eigh(mat)
        eigs = eigs + 1e-16
        sqrt_eigs = torch.sqrt(torch.abs(eigs))
        sqrt_mat = torch.einsum('nij,nj,njk->nik', vecs, sqrt_eigs, vecs.transpose(-2, -1).conj())
        return sqrt_mat

    def forward(self, outputs, targets, step, params=None, rho_Ts = None, glob_loss = False):
        if self.fidelity_type == 'normal':
            outputs_sqrt = sqrt_newton_schulz_autograd(outputs)
            targets_sqrt = sqrt_newton_schulz_autograd(targets)
            avg_fid1 = 1. - torch.mean(torch.sum(torch.sqrt(torch.max(eps, torch.linalg.eigvalsh(contract('mij,njk,mkl->mnil', outputs_sqrt, outputs, outputs_sqrt)))), axis = -1))
            avg_fid2 = 1. - torch.mean(torch.sum(torch.sqrt(torch.max(eps, torch.linalg.eigvalsh(contract('mij,njk,mkl->mnil', targets_sqrt, targets, targets_sqrt)))), axis = -1))
            avg_fid3 = 1. - torch.mean(torch.sum(torch.sqrt(torch.max(eps, torch.linalg.eigvalsh(contract('mij,njk,mkl->mnil', outputs_sqrt, targets, outputs_sqrt)))), axis = -1))

            avg_fid = 2*avg_fid3 - avg_fid1 - avg_fid2

            if glob_loss:
                print(f"[global loss] step {step}, avg_fid1: {avg_fid1}, avg_fid2: {avg_fid2}, avg_fid3: {avg_fid3}, avg_fid: {avg_fid}")
            else:
                print(f"step: {step}, avg_fid1: {avg_fid1}, avg_fid2: {avg_fid2}, avg_fid3: {avg_fid3}, avg_fid: {avg_fid}")
        else:
            # use superfidelity (upper bound)
            # tr(rho sigma) + sqrt(1-tr(rho*rho))*sqrt(1-tr(sigma*sigma))
            outputs_square = contract('mij,mjk->mik', outputs, outputs)
            targets_square = contract('mij,mjk->mik', targets, targets)
            for i in range(outputs.shape[-1]):
                outputs_square[:,i,i] = torch.real(outputs_square[:, i, i])
                targets_square[:,i,i] = torch.real(targets_square[:, i, i])
            outputs_trace = torch.sqrt(torch.max(torch.tensor([1e-16]), 1. - torch.sum(torch.real(torch.diagonal(outputs_square, dim1=-2, dim2=-1)), axis=-1)))
            targets_trace = torch.sqrt(torch.max(torch.tensor([1e-16]), 1. - torch.sum(torch.real(torch.diagonal(targets_square, dim1=-2, dim2=-1)), axis=-1)))
            avg_fid1 = 1. - torch.mean(torch.sum(torch.real(torch.diagonal(contract('mij,njk->mnik', outputs, outputs), dim1=-2, dim2=-1)), axis=-1) + contract('m,n->mn', outputs_trace, outputs_trace))
            avg_fid2 = 1. - torch.mean(torch.sum(torch.real(torch.diagonal(contract('mij,njk->mnik', targets, targets), dim1=-2, dim2=-1)), axis=-1) + contract('m,n->mn', targets_trace, targets_trace))
            avg_fid3 = 1. - torch.mean(torch.sum(torch.real(torch.diagonal(contract('mij,njk->mnik', outputs, targets), dim1=-2, dim2=-1)), axis=-1) + contract('m,n->mn', outputs_trace, targets_trace))
    
            avg_fid = 2 * avg_fid3 - avg_fid1 - avg_fid2

            if glob_loss:
                print(f"[global loss] step {step}, avg_fid1: {avg_fid1}, avg_fid2: {avg_fid2}, avg_fid3: {avg_fid3}, avg_fid: {avg_fid}")
            else:
                print(f"step: {step}, avg_fid1: {avg_fid1}, avg_fid2: {avg_fid2}, avg_fid3: {avg_fid3}, avg_fid: {avg_fid}")
        
        return avg_fid.unsqueeze(0)

class WassersteinLoss(nn.Module):
    def __init__(self, fidelity_type = 'super'):
        super(WassersteinLoss, self).__init__()
        self.fidelity_type = fidelity_type

    def forward(self, outputs, targets, step, err=False, params=None, rho_Ts = None, glob_loss = False):
        '''
        calculate the Wasserstein distance between two sets of density matrices
        the cost matrix is (super) fidelity between density matrices outputs, targets
        '''
        if self.fidelity_type == 'super':
            outputs_square = contract('mij,mjk->mik', outputs, outputs)
            targets_square = contract('mij,mjk->mik', targets, targets)
            outputs_trace = torch.sqrt(torch.max(eps, 1. - torch.sum(torch.real(torch.diagonal(outputs_square, dim1=-2, dim2=-1)), axis=-1)))
            targets_trace = torch.sqrt(torch.max(eps, 1. - torch.sum(torch.real(torch.diagonal(targets_square, dim1=-2, dim2=-1)), axis=-1)))
            D = 1. - (torch.sum(torch.real(torch.diagonal(contract('mij,njk->mnik', outputs, targets), dim1=-2, dim2=-1)), axis=-1) 
                      + contract('m,n->mn', outputs_trace, targets_trace))
        else:
            targets_sqrt = sqrt_newton_schulz_autograd(targets)
            D = 1. - (torch.sum(torch.sqrt(torch.max(eps, torch.linalg.eigvalsh(contract('mij,njk,mkl->mnil', targets_sqrt, outputs, targets_sqrt)))), axis = -1))
        emt = torch.empty(0)
        wasserstein_distance = ot.emd2(emt, emt, M=D)

        if glob_loss:
            print(f"[global loss] step {step}, loss: {wasserstein_distance}")
        else:
            print(f"step: {step}, loss: {wasserstein_distance}")
        
        return wasserstein_distance
    
# Forward via Newton-Schulz iterations
# Backward via autograd
# Source based on: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py

# Compute error
def compute_error(mat, mat_sqrt):
    norm_mat = torch.sqrt(torch.sum(torch.sum(mat.mul(mat.conj()), dim=1),dim=1))
    error = mat - torch.bmm(mat_sqrt, mat_sqrt)
    error = torch.sqrt((error * error).sum(dim=1).sum(dim=1)) / norm_mat
    return error

def sqrt_newton_schulz_autograd(mat: torch.Tensor, num_iters = 10):
    batch_size = mat.shape[0]
    dim = mat.shape[1]
    norm_mat = mat.mul(mat.conj()).sum(dim=1).sum(dim=1).sqrt()
    Y = mat.div(norm_mat.view(batch_size, 1, 1).expand_as(mat));
    I = torch.eye(dim, dim, requires_grad=False).cfloat().reshape(1, dim, dim).repeat(batch_size,1,1)
    Z = torch.eye(dim, dim ,requires_grad=False).cfloat().reshape(1, dim, dim).repeat(batch_size,1,1)

    sqrt_mat = torch.empty_like(mat)
    
    for _ in range(num_iters):
        TT = 0.5*(3.0 * I - Z.bmm(Y))
        Y = Y.bmm(TT)
        Z = TT.bmm(Z)
    
        sqrt_mat_tmp = Y*torch.sqrt(norm_mat).view(batch_size, 1, 1).expand_as(mat)
        if torch.any(torch.isnan(sqrt_mat_tmp)):
            break

        sqrt_mat = sqrt_mat_tmp

        error = compute_error(mat, sqrt_mat)
        if torch.all(torch.isclose(error, torch.zeros_like(error).cfloat(), atol=1e-5)):
            break
    
    return sqrt_mat
