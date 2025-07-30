# MSQuDDPM
The official Python implementation of the [Mixed-State Quantum Denoising Diffusion Probabilistic Model](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.032610) by [Gino Kwun](https://staphaniek.github.io/aboutme/), [Dr. Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), and [Dr. Quntao Zhuang](https://viterbi.usc.edu/directory/faculty/Zhuang/Quntao).

# Dependencies

To easily run the code, please install the dependencies by executing the following command in the terminal from the root directory:

```
pip install -r requirements.txt
```


This implementation is based on Python 3.11.7 and utilizes [Tensorcircuit](https://tensorcircuit.readthedocs.io/en/latest/#) and [PyTorch](https://pytorch.org/). [POT](https://pythonot.github.io/) is used to calculate the Wasserstein distance, and [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) is employed for advanced matrix computation. We also leverage the [QuTip](https://qutip.org/) package for visualizing quantum data on the Bloch sphere and generating random states.

The exact versions of all required libraries are listed in the `requirements.txt` file.

# Training


To train the MSQuDDPM model, please run `src/train.py`. For instance, to train on the cluster state generation task, use the following command:

```
python src/train.py --initial_type cluster --T 4 --n_data 1000 --n_train 100 --n_ancilla 1 --L 6 --xavier_init false --use_zero_ancillas false --p_limit 1.0 --beta_schedule cosine --beta_start 0.1 --beta_end 1. --lr 7e-3 --lr_decay_rate 0.5 --lr_decay_count 4 --fidelity_type super --loss_type mmd --max_epoch 2001 --global_loss true
```

The --initial_type argument specifies the generation task, and supports three options: cluster, circular, and many_body_phase. You can freely modify the configuration to experiment with different training setups.

Descriptions of command-line arguments can be found in `src/arguments.py`.

After training, model parameters can be saved using functions such as `np.save()`. This can be added at the end of the main() function in `src/train.py` for later use.


# Generation

To generate samples using a trained or pre-trained MSQuDDPM model, please use the `prepare_backward_states()` function in `src/models/backward_denoising_model.py`. Sample generation is handled by `src/generation.py`. For instance:

```
python src/generation.py --initial_type cluster --T 4 --n_data 1000 --n_train 100 --n_ancilla 1 --L 6 --xavier_init false --use_zero_ancillas false --p_limit 1.0 --beta_schedule cosine --beta_start 0.1 --beta_end 1. --lr 7e-3 --lr_decay_rate 0.5 --lr_decay_count 4 --fidelity_type super --loss_type mmd --max_epoch 2001 --global_loss true
```

Before running, please make sure to either define appropriate opt_params or load the trained parameters directly in `src/generation.py`.

# Visualization

The notebook `MSQuDDPM_Visualizations.ipynb` provides a simple guide for visualizing simulation results. Training hyperparameters for different generation tasks are also included in the notebook.

# Citation

If you find this implementation helpful, we kindly ask you to cite the following work:

```
@article{kwun2025msquddpm,
  title={Mixed-state quantum denoising diffusion probabilistic model},
  author={Kwun, Gino and Zhang, Bingzhi and Zhuang, Quntao},
  journal={Physical Review A},
  volume={111},
  number={3},
  pages={032610},
  year={2025},
  publisher={APS}
}
```