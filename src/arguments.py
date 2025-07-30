import argparse

def str2bool(v):
    return v.lower() == 'true'

def get_args(arg_str=None):
    parser = argparse.ArgumentParser(description='QuDDPM')

    parser.add_argument('--initial_type', type=str, default='cluster', help='generation task type (cluster, ring, many_body_phase)')
    parser.add_argument('--L', type=int, default=4, help='Number of layers in the single backward curcuit')
    parser.add_argument('--max_epoch', type=int, default=2001, help='Maximum iterations in training')
    parser.add_argument('--lr', type=float, default=5e-3,
                    help='learning rate for Adam optimizer (default: 5e-3).')
    parser.add_argument('--lr_decay_rate', type=float, default = 1.0, help='learning rate decay ratio (default: 1.).')
    parser.add_argument('--lr_decay_count', type=int, default = 2, help='learning rate decay count (default: 2). decays lr_decay_count - 1 time(s)')
    parser.add_argument('--xavier_init', type=str2bool, default=False)
    parser.add_argument('--use_zero_ancillas', type=str2bool, default=False)
    parser.add_argument('--fidelity_type', type=str, default='super', help='fidelity type in loss function (super, normal)')
    parser.add_argument('--loss_type', type=str, default='mmd')
    parser.add_argument('--n_data', type=int, default=1000)
    parser.add_argument('--n_train', type=int, default=100)
    parser.add_argument('--T', type=int, default=20, help='Number of stages in MSQuDDPM model')
    parser.add_argument('--n_ancilla', type=int, default=2, help='Number of ancillas')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='Forward scheduling type. (linear, cosine, sq_cosine)')
    parser.add_argument('--beta_start', type=float, default = -1)
    parser.add_argument('--beta_end', type=float, default = -1)
    parser.add_argument('--global_loss', type=str2bool, default=False)

    args = parser.parse_args()

    return args
