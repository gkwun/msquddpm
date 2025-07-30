# Common
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
# rc('text', usetex=False)
rc('axes', linewidth=1)

from opt_einsum import contract

# Qutip
import qutip as qt
from qutip import Bloch, rand_ket_haar

# Tensorcircuit
import tensorcircuit as tc

K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

# Pytorch
import torch

def show_states(states):
    expect_xs, expect_ys, expect_zs = get_bloch_positions(states)
    plot_coordinates(expect_xs, expect_ys, expect_zs)

def get_bloch_positions(states):
    """
    Get bloch sphere position (x, y, z) of the quantum states

    Args:
    states: list of single quantum states
    """
    sigmas = [qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()]
    psis = contract('mi,mj->mij', states, states.conj())
    positions = [np.real(contract('mii->m', contract('mij,jk->mik', psis, sigma))) for sigma in sigmas]
    return positions

def trace_out_np(density_matrices):
    if density_matrices.shape[-1] == 2:
        return density_matrices

    if len(density_matrices.shape) == 4:
        # t, n, i, j
        t, size, _, dim = density_matrices.shape
        result = np.zeros((t, size, 2, 2), dtype = np.complex64)
        dim = dim // 2
        for i in range(dim):
            idx_operator = np.zeros(dim, dtype = np.complex64)
            idx_operator[i] = 1.
            trace_3qubit_operator = np.kron(np.eye(2), idx_operator)
            result += np.einsum('ij,tnjk,kl->tnil', trace_3qubit_operator, density_matrices, trace_3qubit_operator.T)
    elif len(density_matrices.shape) == 3:
        # n, i, j
        size, _, dim = density_matrices.shape
        result = np.zeros((size, 2, 2), dtype = np.complex64)
        dim = dim // 2
        for i in range(dim):
            idx_operator = np.zeros(dim, dtype = np.complex64)
            idx_operator[i] = 1.
            trace_3qubit_operator = np.kron(np.eye(2), idx_operator)
            result += np.einsum('ij,njk,kl->nil', trace_3qubit_operator, density_matrices, trace_3qubit_operator.T)

    return result

def show_density_matrices(density_matrices, colors = None):
    density_matrices = trace_out_np(density_matrices)
    expect_xs, expect_ys, expect_zs = get_density_matrix_bloch_positions(density_matrices)
    plot_coordinates(expect_xs, expect_ys, expect_zs, colors)

def get_density_matrix_bloch_positions(density_matrices):
    """
    Get bloch spehre position (x, y, z) of the density matrices
    
    Args:
    density_matrices: Collection of single 2 x 2 density matrices
    """
    a = density_matrices[:, 0, 0]
    b = density_matrices[:, 1, 0]
    xs = 2.0 * b.real
    ys = 2.0 * b.imag
    zs = 2.0 * a - 1.0
    return xs, ys, zs

def plot_coordinates(xs, ys, zs, colors = None):
    fig, axs = plt.subplots(1, 1, figsize=(4,4), subplot_kw={'projection': '3d'})
    b = Bloch(fig=fig, axes=axs)
    b.clear()
    # b.figsize = [3, 3]
    if colors != None:
        b.add_points([xs, ys, zs], 'm')
        b.point_color = colors
    else:
        b.add_points([xs, ys, zs])
    b.point_size = 8*np.ones(len(xs))
    b.render()
    plt.tight_layout()
    
def show_forward_outputs(forward_outputs, fig, axs, T, n_data):
    """
    plot forward outputs
    Args:
    forward_outputs: [T+1, n_data, 2**n, 2**n] forward outputs.
    fig, axs: pyplot figure and axis
    """
    forward_outputs = trace_out_np(forward_outputs)
    col = 0
    if T < 4:
        step = 1
    elif T % 3 == 0:
        step = T // 3
    else:
        step = T//4
    for t in range(0, T+1, step):
        xs, ys, zs = get_density_matrix_bloch_positions(forward_outputs[t])
        b = Bloch(fig=fig, axes=axs[col])
        b.clear()
        b.add_points([xs, ys, zs], 'm')
        b.point_color = ['b']*(forward_outputs[t].shape[0])
        b.point_size = 8 * np.ones(n_data)
        b.font_size = 23
        b.render()
        axs[col].set_title(r'$t=%d$'%t, fontsize=30)
        col+=1
    plt.tight_layout()

def show_backward_outputs(backward_outputs, fig, axs, T, n_train):
    """
    plot forward outputs
    Args:
    forward_outputs: [T+1, n_data, 2**n, 2**n] forward outputs.
    fig, axs: pyplot figure and axis
    """
    backward_outputs = trace_out_np(backward_outputs)
    col = 0
    if T < 4:
        step = 1
    elif T % 3 == 0:
        step = T // 3
    else:
        step = T//4
    for t in range(0, T+1, step):
        xs, ys, zs = get_density_matrix_bloch_positions(backward_outputs[t])
        b = Bloch(fig=fig, axes=axs[col])
        b.clear()
        b.add_points([xs, ys, zs], 'm')
        b.point_color = ['r']*(backward_outputs[t].shape[0])
        b.point_size = 8*np.ones(n_train)
        b.font_size = 23
        b.render()
        axs[col].set_title(r'$t=%d$'%t, fontsize=30)
        col+=1
    plt.tight_layout()

def get_purity(x):
    return np.mean(np.trace(np.matmul(x, x), axis1=-2, axis2=-1))

def get_purities(model_outputs):
    T, _, _, _ = model_outputs.shape
    T -= 1
    purities = []
    for t in range(T+1):
        purities.append(get_purity(model_outputs[t]))
    return purities

def get_cluster(n, n_data, scale = 0.04, seed=None):
    '''
    Generate random quantum states in state vector format close to |0...0>
    Args:
    n: number of qubits
    n_data: number of samples to generate
    scale: the scaling factor on amplitudes except |0...0>
    seed: control the randomness
    '''
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    # amplitude for basis except |0...0>
    remains = np.random.randn(n_data, (2**n) - 1) + 1j * np.random.randn(n_data, (2**n) - 1) 
    states = np.hstack((np.ones((n_data, 1)), scale * remains)) # un-normalized
    states /= np.tile(np.linalg.norm(states, axis=1).reshape((1, n_data)), (2**n, 1)).T
    return states.astype(np.complex64)

def small_depolarize(n, init_state, scale = 0.01, seed=None):
    n_dim = 2**n
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    result = np.outer(init_state, init_state.conj())
    for i in range(2**n):
        result[i,i] = result[i,i].real
    for i in range(n):
        p = 1. * np.random.rand() * scale # Uniform(0, 0.01)
        result = (1-p) * result + p / n_dim * np.eye(n_dim)

    return result

def get_initial_states(type, n, n_data, epsilon = 0.04):
    init_dms = np.zeros((n_data, 2**n, 2**n), dtype=np.complex64)
    if type == 'cluster':
        init_states = get_cluster(n, n_data, epsilon)
        for i in range(n_data):
            init_dms[i] = small_depolarize(n, init_states[i], scale=0.01)
    elif type == 'many_body_phase':
        glist_train = np.random.uniform(1.8, 2.2, size=n_data)

        init_states = []
        for g in glist_train:
            _, vecs = np.linalg.eigh(get_tfim(n, g))
            init_states.append(vecs[:, 0])
        init_states = np.stack(init_states)
        for i in range(n_data):
            init_dms[i] = np.outer(init_states[i], init_states[i].conj())
    elif type == 'circular':
        phis = np.random.uniform(0, 2 * np.pi, n_data)
        init_states = np.vstack((np.cos(phis), np.sin(phis))).astype(np.complex64).T
        for i in range(n_data):
            init_dms[i] = small_depolarize(n, init_states[i], scale=epsilon)
    else:
        raise Exception('type is wrong. should be cluster/circular/many_body_phase.')
    
    return init_dms

def get_tfim(n, g):
    # hamiltonian matrix of tfim
    x = np.array([[0, 1], [1, 0]])
    z = np.array([[1,0], [0, -1]])
    zz = np.kron(z, z)
    h = 0
    for i in range(n-1):
        h -= np.kron(np.kron(np.eye(2**i), zz), np.eye(2**(n-2-i)))
    for i in range(n):
        h -= g* np.kron(np.kron(np.eye(2**i), x), np.eye(2**(n-1-i)))
    return h

def get_magnetization(density_matrix, n):
    """
    Return X axis magnetization.
    """
    M = 0.
    xxx = np.zeros((2**n, 2**n))
    x = np.array([[0,1], [1, 0]])
    for ii in range(n):
        x_i = np.kron(np.eye(2**ii), np.kron(x, np.eye(2**(n-ii-1))))
        xxx += x_i
    M = np.trace(np.matmul(density_matrix, xxx) / n)
    return M

def get_plot_format(T):
    if T % 3 == 0:
        return plt.subplots(1, 4, figsize=(14,10), subplot_kw={'projection': '3d'})
    else:
        return plt.subplots(1, 5, figsize=(14,10), subplot_kw={'projection': '3d'})

def plot_forward_process(forward_model, n, n_data, n_train, T, init_dms):
    forward_outputs = np.zeros((T+1, n_train, 2**n, 2**n), dtype=np.complex64)

    for t in range(0, T+1):
        indices = np.sort(np.random.choice(n_data, size=n_train, replace=False))
        forward_outputs[t] = forward_model.get_forward_density_matrices(t, init_dms)[indices]

    fig, axs = get_plot_format(T)

    show_forward_outputs(forward_outputs, fig, axs, T, n_data)

def plot_forward_process_outputs(forward_outputs, n_data, T):
    fig, axs = get_plot_format(T)

    show_forward_outputs(forward_outputs, fig, axs, T, n_data)

def plot_backward_process(model, opt_params, inputs_backward, n, T, n_train):
    backward_outputs = np.zeros((T+1, n_train, 2**n, 2**n), dtype=np.complex64)

    for t in range(T, -1, -1):
        backward_outputs[t] = model.prepare_backward_states(t, inputs_backward, opt_params)[:, :2**n, :2**n]

    fig, axs = get_plot_format(T)

    show_backward_outputs(backward_outputs, fig, axs, T, n_train)

def plot_backward_process_outputs(backward_outputs, T, n_train):
    fig, axs = get_plot_format(T)

    show_backward_outputs(backward_outputs, fig, axs, T, n_train)

def plot_loss_history(loss_histories):
    # plot training loss history
    fig, axs = plt.subplots((T+3)//4, 4, figsize=(16, 14 / 5 * ((T+3)//4)), sharex=True, sharey=True)
    loss_hist = loss_histories.detach().numpy()
    for i in range(T):
        if (T+3)//4 == 1:
            axs[i%4].plot(loss_hist[i+1], lw=2)
            axs[i%4].tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=20)
            axs[i%4].text(x=680, y=0.4, s=r'$t=%d$'%(i+1), fontsize=25)
            axs[i%4].set_yscale('log')
        else:
            axs[i//4, i%4].plot(loss_hist[i+1], lw=2)
            axs[i//4, i%4].tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=20)
            axs[i//4, i%4].text(x=680, y=0.4, s=r'$t=%d$'%(i+1), fontsize=25)
            axs[i//4, i%4].set_yscale('log')
    fig.supxlabel(r'$\rm iterations$', fontsize=30)
    fig.supylabel(r'$\mathcal{L}(t)$', fontsize=30)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_global_loss_history(fig, axs, l_histories, param_size):
    # plot training loss history
    loss_hist = np.array(l_histories)
    axs.plot(range(loss_hist.shape[0]),loss_hist, '-', c='blue', mfc='white', markersize=8, label=r'${\rm forward}$')
    axs.tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=20)
    fig.supxlabel(r'$\rm time (10 sec)$', fontsize=30)
    fig.supylabel(r'$\mathcal{L}(\{\widetilde{\rho}_t\}, \{\rho_0\})$', fontsize=30)
    # plt.tight_layout()
    plt.show()
    plt.close()

def plot_global_loss_history_iter(fig, axs, l_histories, param_size, l_histories2 = None, param_size2 = None):
    # plot training loss history
    loss_hist = np.array(l_histories)
    loss_hist_x = np.arange(0, len(l_histories) * param_size, param_size)
    axs.plot(loss_hist_x,loss_hist, '-', c='gray', mfc='white', markersize=8, linewidth=1, label=r'${\rm T=2,L=21,n_a = 6}$')
    if l_histories2 is not None:
        loss_hist = np.array(l_histories2)
        loss_hist_x = np.arange(0, len(l_histories2) * param_size2, param_size2)
        axs.plot(loss_hist_x,loss_hist, '-', c='purple', mfc='white', markersize=8, linewidth=1, label=r'${\rm T=6,L=12,n_a = 2}$')
    
    axs.legend(fontsize=20, framealpha=0)
    axs.tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=20)
    fig.supxlabel(r'$\text{epochs} \times \text{single step parameters}$', fontsize=20)
    fig.supylabel(r'$\rm D_{MMD}(\{\widetilde{\rho}_t\}, \{\rho_0\})$', fontsize=20)
    x_major_locator = MultipleLocator(20000)
    axs.xaxis.set_major_locator(x_major_locator)
    plt.tight_layout()
    plt.show()
    plt.close()

def compare_fw_scheduling(fig, axs, l_histories, param_size, l_histories2 = None, param_size2 = None, l_histories3 = None, param_size3 = None):
    # plot training loss history
    loss_hist = np.array(l_histories)
    loss_hist_x = np.arange(0, len(l_histories) * param_size, param_size)
    axs.plot(loss_hist_x,loss_hist, '--', c='gray', mfc='white', markersize=0, linewidth=1, label=r'${\rm linear}$')
    if l_histories2 is not None:
        loss_hist = np.array(l_histories2)
        loss_hist_x = np.arange(0, len(l_histories2) * param_size2, param_size2)
        axs.plot(loss_hist_x,loss_hist, '--', c='green', mfc='white', markersize=0, linewidth=1, label=r'${\rm cosine}$')
    if l_histories3 is not None:
        loss_hist = np.array(l_histories3)
        loss_hist_x = np.arange(0, len(l_histories3) * param_size3, param_size3)
        axs.plot(loss_hist_x,loss_hist, '--', c='purple', mfc='white', markersize=0, linewidth=1, label=r'${\rm sq_ cosine}$')
    
    axs.legend(fontsize=20, framealpha=0)
    axs.tick_params(direction='in', length=6, width=2, top='on', right='on', labelsize=23)
    fig.supxlabel(r'$\rm iterations $', fontsize=20)
    fig.supylabel(r'$D_{\rm MMD}(\{\tilde{\rho}_t\}, \{\rho_0\})$', fontsize=20)
    plt.show()
    plt.close()

def print_train_detail(jobID, T, initial_type, beta_schedule, beta_start, beta_end, max_epoch, L, n_data, n_train, n_ancilla, lr, lr_decay_rate, lr_decay_count, xavier_init, use_zero_ancillas, fidelity_type, loss_type):
    print(f"jobID: {jobID}, T: {T}, initial_type: {initial_type}, beta_schedule: {beta_schedule}, (beta_start, beta_end) = ({beta_start}, {beta_end}), max_epoch: {max_epoch}, L: {L}, n_data: {n_data}, n_train: {n_train}, n_ancilla: {n_ancilla}, lr: {lr}, decay {lr_decay_rate} {lr_decay_count-1} times, xavier_init: {xavier_init}, use_zero_ancillas: {use_zero_ancillas}, fidelity_type: {fidelity_type}, loss_type: {loss_type}")

def get_haar_states(n, n_data, seed = None):
    dims = 2 ** n
    haar_states = np.asarray([rand_ket_haar(dims, seed = seed).full().flatten() for _ in range(n_data)], dtype=np.complex64)
    return haar_states

def get_haar_density_matrices(n, n_data, seed = None):
    haar_states = get_haar_states(n, n_data, seed)
    haar_dms = np.asarray([np.outer(state, state.conj()) for state in haar_states], dtype=np.complex64)
    return haar_dms

def plot_purity(T, forward_outputs = None, backward_outputs = None):
    fig, ax = plt.subplots(figsize=(5, 3))
    if forward_outputs is not None:
        forward_purities = np.array(get_purities(forward_outputs))
        ax.plot(range(T+1), forward_purities, 'o--', lw=1, c='b')
    if backward_outputs is not None:
        backward_purities = np.array(get_purities(backward_outputs))
        ax.plot(range(T+1), backward_purities, 'o-', lw=1, c='r')
    ax.set_ylabel(r'${\text{Mean Purity}}$', fontsize=20)
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylim(0.4,1)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=20)

def plot_wass_distance(T, forward_global_losses = None, backward_global_losses = None):
    fig, ax = plt.subplots(figsize=(5, 3))
    if forward_global_losses is not None:
        ax.plot(range(T+1), forward_global_losses, 'o--', lw=1, c='b')
    if backward_global_losses is not None:
        ax.plot(range(T+1), backward_global_losses, 'o-', lw=1, c='r')
    ax.set_ylabel(r'$Wass (\{\widetilde{\rho}_t\}, \{\rho_0\})$', fontsize=20)
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylim(0,0.6)
    ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=20)
