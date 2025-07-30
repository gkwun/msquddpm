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

import time

import utils
import loss_functions as lf
from models.forward_diffusion_model import ForwardDiffusionModel
from models.backward_denoising_model import BackwardDenoisingModel

def train(model, t, inputs_backward, prev_params, max_epoch, n_train, lr = 7e-3, lr_decay_rate = 1.0, xavier_init = False, lr_decay_count=4, fidelity_type='super', loss_type='mmd', compute_global_loss = False, initial_forward = None):
    '''
    Trianing for the backward PQC at step t
    Args:
    inputs_backward: backward input at step T
    model: QDDPM model
    t: diffusion step
    prev_params: collection of PQC parameters before step t
    max_epoch: the number of iterations
    n_train: training data size
    lr: learning rate
    lr_decay_rate: decay rate of learning rate
    lr_decay_count: the number of learning rate decay in exponential decay format
    xavier_init: if True, use xavier initialization for parameters
    fidelity_type: fidelity type in loss computation
    loss_type: loss function type
    compute_global_loss: if True, compute global loss with initial forward inputs
    '''
    n = model.n
    n_ancilla = model.n_ancilla
    L = model.L
    loss_fn = lf.get_loss_fn(loss_type, fidelity_type)
    # For global loss calculation with forward input at t = 0
    loss_fn_global = lf.get_loss_fn(loss_type, fidelity_type)
    
    # Form the inputs for step t, which is the outputs of step t+1
    psi_tplus1_tildas = model.prepare_backward_states(t, inputs_backward, prev_params)
    tmp = np.zeros_like((len(psi_tplus1_tildas), 2**n, 2**n))
    tmp = psi_tplus1_tildas.numpy()[:,:2**n, :2**n]

    print(f"t={t} input data")
    utils.show_density_matrices(tmp)
    print(f"input data purity: {utils.get_purity(tmp)}")
    target_states = model.target_states[t-1]
    loss_hist = []
    global_loss_hist = []

    # Initialize parameters (normal and xavier initialization)
    if xavier_init:
        params = torch.tensor(np.random.normal(scale = 1. / np.sqrt(n + n_ancilla), size=2 * (n + n_ancilla) * L))
    else:
        params = torch.tensor(np.random.normal(scale = 1., size=2 * (n + n_ancilla) * L))
    # Initialize ancillary qubits to normal for diversity of projective measurement results
    for i in range(params.shape[0]):
        if i % (2 * (n + n_ancilla)) < 2 * n_ancilla:
            params[i] = torch.tensor(np.random.normal(scale = 1., size=1))
    params.requires_grad = True
    
    print(f"[t={t}] initial params: {params}")
    # Haar ancillas for the step if we use Haar ancillas
    haar_ancillas = utils.get_haar_density_matrices(1, n_train)
    haars = torch.from_numpy(haar_ancillas).cfloat()
    
    # Set optimizer
    optimizer = torch.optim.Adam([params], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay_rate)
    
    rho_Ts = inputs_backward
    
    t0 = time.time()
    for step in range(max_epoch):
        indices = np.random.choice(len(target_states), size=n_train, replace=False)
        psi_ts = target_states[indices]
        output_t = model.forward(psi_tplus1_tildas, params, haars)
        optimizer.zero_grad()

        loss = loss_fn.forward(output_t, psi_ts, step, params = params, rho_Ts = rho_Ts)

        try:
            loss.backward()
        except Exception as e:
            print(e)
            break
        
        optimizer.step()

        loss_hist.append(loss)
        elapsed_time_sec = time.time() - t0
        if compute_global_loss:
            global_loss = loss_fn_global.forward(output_t, initial_forward, step, params = params, rho_Ts = None, glob_loss = True)
            global_loss_hist.append(global_loss)
        
        if step%400 == 0 or step == max_epoch-1:
            print(f"[t={t}, step {step}] output_t")
            utils.show_density_matrices(output_t.detach().numpy())
            print("psi_ts")
            utils.show_density_matrices(psi_ts.numpy())
            print("Step %s, loss: %s, time elapsed: %s seconds"%(step, loss, elapsed_time_sec))
        if (step+1)%(max_epoch//lr_decay_count) == 0 and lr_decay_rate != 1.0 and step + 2 < max_epoch:
            lr_scheduler.step()
            print(f"lr decay by {lr_decay_rate}; new lr: {optimizer.state_dict()['param_groups'][0]['lr']}")

    return params, torch.stack(loss_hist).reshape(-1), torch.stack(global_loss_hist).reshape(-1)

def main():
    # Set parameters from arguements
    args = get_args()

    seed = None
    
    initial_type = args.initial_type
    if initial_type in ['many_body_phase']:
        n = 4
    else:
        n = 1

    L = args.L
    lr = args.lr
    lr_decay_rate = args.lr_decay_rate
    lr_decay_count = args.lr_decay_count
    fidelity_type = args.fidelity_type
    xavier_init = args.xavier_init
    use_zero_ancillas = args.use_zero_ancillas
    loss_type = args.loss_type
    max_epoch =args.max_epoch
    n_data = args.n_data
    n_train = args.n_train
    T = args.T
    n_ancilla = args.n_ancilla
    beta_schedule = args.beta_schedule
    beta_start = args.beta_start
    beta_end = args.beta_end
    compute_global_loss = args.global_loss
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
    backward_model.set_target_states(forward_outputs)

    # Global loss computation with forward inputs at t = 0
    global_losses = []
    idxs = np.random.choice(n_data, size=n_train, replace=False)
    initial_forward = torch.from_numpy(forward_outputs[0, idxs])
    
    opt_params = []
    loss_histories = torch.zeros((T+1, max_epoch))

    startpoint = T - len(opt_params)

    print(torch.cuda.is_available())
    utils.print_train_detail(None, T, initial_type, beta_schedule, beta_start, beta_end, max_epoch, L, n_data, n_train, n_ancilla, lr, lr_decay_rate, lr_decay_count, xavier_init, use_zero_ancillas, fidelity_type, loss_type)
    
    for t in range(startpoint, 0, -1):
        params, loss_histories[t], global_loss_hist = train(backward_model, t, inputs_backward, np.asarray(opt_params), max_epoch, n_train=n_train, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_count=lr_decay_count, xavier_init=xavier_init, fidelity_type = fidelity_type, loss_type = loss_type, compute_global_loss = compute_global_loss, initial_forward=initial_forward)
        p = params.detach().numpy().tolist()
        opt_params.append(p)
        print(f"opt_params: {opt_params}")
        if compute_global_loss:
            gloss = global_loss_hist.detach().numpy().tolist()
            global_losses.append(gloss)
            print(f"global_losses: {global_losses}")
        print("loss_histories:")
        for tt in range(0, T+1):
            print(f"{loss_histories[tt].detach().numpy().tolist()},")

    print("Train finished!")
    

if __name__ == '__main__':
    main()
