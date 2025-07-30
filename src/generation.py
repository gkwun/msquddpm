# Argument Parsing
from arguments import get_args

# Common
import numpy as np

# Tensorcircuit
import tensorcircuit as tc

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

# Pytorch
import torch

import utils
import loss_functions as lf
from models.forward_diffusion_model import ForwardDiffusionModel
from models.backward_denoising_model import BackwardDenoisingModel


def main():
    # Set parameters from arguements
    args = get_args()

    # Set opt_params by directly assigning the values or load the trained parameters with np.load() function
    # Remember to set appropriate values for L, T, n, n_ancilla when running the code.
    opt_params = []
    # opt_params = np.load('data/cluster/T6L4WassHaarCosine_optParams.npy').tolist()

    seed = None
    
    initial_type = args.initial_type
    if initial_type in ['many_body_phase']:
        n = 4
    else:
        n = 1

    L = args.L
    use_zero_ancillas = args.use_zero_ancillas
    n_data = args.n_data
    n_train = args.n_train
    T = args.T
    n_ancilla = args.n_ancilla
    beta_schedule = args.beta_schedule
    beta_start = args.beta_start
    beta_end = args.beta_end
    if beta_start < 0:
        beta_start = None
        beta_end = None
    epsilon = 0.04

    init_dms = utils.get_initial_states(initial_type, n, n_data, epsilon)
    dims = 2 ** n
    # Totally mixed states for initial backward inputs
    rho_T_tildas = np.asarray([1. / dims * np.eye(dims) for _ in range(n_data)], dtype = np.complex64)

    forward_model = ForwardDiffusionModel(n, T, n_data, beta_schedule = beta_schedule, beta_start = beta_start, beta_end = beta_end)
    forward_outputs = np.zeros((T+1, n_data, 2**n, 2**n), dtype=np.complex64)

    for t in range(0, T+1):
        forward_outputs[t] = forward_model.get_forward_density_matrices(t, init_dms, seed=seed)
    
    # Always use totally mixed states for backward inputs
    inputs_backward = torch.from_numpy(rho_T_tildas[:n_train]).cfloat()

    backward_model = BackwardDenoisingModel(n, n_ancilla, T, L, use_zero_ancillas)
    backward_outputs = np.zeros((T+1, n_train, 2**n, 2**n), dtype=np.complex64)

    for t in range(T, -1, -1):
        backward_outputs[t] = backward_model.prepare_backward_states(t, inputs_backward, opt_params)[:, :2**n, :2**n]
    
    # Please change the file name that meets to your purpose
    np.save('task_%s_n%dna%dT%dL%d_schedule_%s_forward_outputs.npy'%(initial_type, n, n_ancilla, T, L, beta_schedule), forward_outputs)
    np.save('task_%s_n%dna%dT%dL%d_schedule_%s_backward_outputs.npy'%(initial_type, n, n_ancilla, T, L, beta_schedule), backward_outputs)

    # To check the outputs, use the code as follows:
    # fw = np.load('task_%s_n%dna%dT%dL%d_schedule_%s_forward_outputs.npy'%(initial_type, n, n_ancilla, T, L, beta_schedule))
    # bw = np.load('task_%s_n%dna%dT%dL%d_schedule_%s_backward_outputs.npy'%(initial_type, n, n_ancilla, T, L, beta_schedule))

if __name__ == '__main__':
    main()
