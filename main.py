import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from config import config
from model import Transformer
from datasets_v2 import *
from definitions import WANDB_KEY

ATTENTION_CMAP = 'inferno'

def eval_loss_and_accuracy(mod, inputs, labels, criterion):
    y_hat, out_dict = mod(inputs, save_weights=config.save_weights)
    loss = criterion(y_hat, torch.argmax(labels.float(), dim=-1))
    predicted_labels = torch.argmax(y_hat, dim=1)
    accuracy = (predicted_labels == torch.argmax(labels.float(), dim=-1)).float().mean()
    return loss, accuracy, out_dict


def set_config(config):
    """The default config arguments can be overridden with command line arguments here.
    """
    parser = argparse.ArgumentParser(description="Run script with overridden configuration.")

    # Add arguments for each configuration setting you want to override
    # model hyperparameters
    parser.add_argument("--h_dim", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--n_blocks", type=int)
    parser.add_argument("--activation", type=str)
    parser.add_argument('--apply_ln', type=int, choices=[1, 0], help="Enable LayerNorm in the model.")
    parser.add_argument("--widening_factor", type=int)
    parser.add_argument("--max_T", type=int)
    parser.add_argument("--drop_p", type=float)
    # data hyperparameters
    parser.add_argument("--S", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--L", type=int)
    parser.add_argument("--D", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--eps", type=float)
    # sequence hyperparameters
    parser.add_argument("--N", type=int)
    parser.add_argument("--B", type=int)
    parser.add_argument("--pB", type=float)
    parser.add_argument("--pC", type=float)
    # training hyperparameters
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--w_decay", type=float)
    parser.add_argument("--niters", type=int)
    # logging
    parser.add_argument("--log_to_wandb", type=bool)  # this doesn't work with argparse,
    parser.add_argument("--logging_interval", type=int)

    args = parser.parse_args()

    # update config with command line arguments
    for key, value in vars(args).items():
        # only update if value is not None
        if value is not None:
            if key == "apply_ln":
                # convert int to boolean
                value = value == 1
            if key in config.model:
                config.model[key] = value
            elif key in config.data:
                config.data[key] = value
            elif key in config.seq:
                config.seq[key] = value
            elif key in config.train:
                config.train[key] = value
            elif key == 'log_to_wandb':
                config.log_to_wandb = value
            elif key == 'logging_interval':
                config.logging_interval = value

    return config


def main(config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # get parameters from config
    # data parameters
    S = config.data.S
    K = config.data.K  # number of classes
    L = config.data.L  # number of labels
    D = config.data.D  # dimension of inputs
    alpha = config.data.alpha  # zipf exponent
    eps = config.data.eps  # within-class variance
    # sequence parameters
    N = config.seq.N
    B = config.seq.B
    pB = config.seq.pB
    pC = config.seq.pC
    Nmax = 32  # this is fixed.
    # determine the frequency of different classes
    p_class = 1.0 / (np.arange(1, K + 1) ** alpha)
    p_class /= np.sum(p_class)

    ln = config.model.apply_ln
    experiment_name = 'I{}_K{}_N{}_L{}_D{}_a{}_B{}_pB{}_pC{}_eps{}_lr{}_drop{}_{}_ln{}_wDecay{}'.format(
        config.train.niters,
        config.data.K,
        config.seq.N,
        config.data.L,
        config.data.D,
        config.data.alpha,
        config.seq.B,
        config.seq.pB,
        config.seq.pC,
        config.data.eps,
        config.train.learning_rate,
        config.model.drop_p,
        ln,
        config.train.w_decay,
        config.train.optim
    )
    config.model.out_dim = config.data.L
    print(experiment_name)
    if config.log_to_wandb:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="IWvsIClearning", name=experiment_name, config=config)
    # Loading datasets
    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)
    test_inputs, test_labels = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps, P=p_class,
                                                   B=B,
                                                   p_B=pB, p_C=pC, no_repeats=True)
    test_inputs_ic, test_labels_ic = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps,
                                                         P=p_class,
                                                         B=B, p_B=1, p_C=1, no_repeats=True)
    test_inputs_iw, test_labels_iw = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps,
                                                         P=p_class,
                                                         B=0, p_B=0, p_C=0, no_repeats=True)
    # cast to torch tensor
    test_inputs_ic = torch.from_numpy(np.array(test_inputs_ic)).float().to(device)
    test_inputs_iw = torch.from_numpy(np.array(test_inputs_iw)).float().to(device)
    test_labels_ic = torch.from_numpy(np.array(test_labels_ic)).to(device)
    test_labels_iw = torch.from_numpy(np.array(test_labels_iw)).to(device)
    test_inputs = torch.from_numpy(np.array(test_inputs)).float().to(device)
    test_labels = torch.from_numpy(np.array(test_labels)).to(device)
    # initialize model, optimizer, loss fn
    model = Transformer(config=config.model).to(device)  # my custom transformer encoder
    if config.train.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate_adam, weight_decay=config.train.w_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.w_decay)
    criterion = nn.CrossEntropyLoss()
    # training loop
    for n in range(config.train.niters):
        model.train()

        # load in a batch of data
        # dataload_start = time.time()
        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label, mus_class, labels_class,
                                                                         config.train.batch_size, N, Nmax,
                                                                         eps=eps, P=p_class, B=B, p_B=pB, p_C=pC,
                                                                         output_target_labels=True, no_repeats=True)
        # cast to torch tensor (TODO: there's gotta be a better way to do this)
        inputs_batch = torch.from_numpy(inputs_batch).float().to(device)
        labels_batch = torch.from_numpy(np.array(labels_batch)).to(device)
        # dataload_end = time.time()
        # print(f'time to load data: {dataload_end-dataload_start}')

        optimizer.zero_grad()
        # forward_pass_start = time.time()
        y_hat, out_dict = model(inputs_batch)
        # forward_pass_end = time.time()
        # print(f'time taken for forward pass: {forward_pass_end-forward_pass_start}')

        # optimizer_start = time.time()
        loss = criterion(y_hat, torch.argmax(labels_batch.float(), dim=-1))
        loss.backward()
        optimizer.step()
        # optimizer_end = time.time()
        # print(f'time taken for backward pass: {optimizer_end-optimizer_start}')

        # evaluate on ICL, IWL etc

        if n % config.logging_interval == 0:
            model.eval()
            with torch.no_grad():
                if config.log_to_wandb:
                    wandb.log({'train_loss': loss.item(), 'iter': n})

                # evaluate on test set (same dist as training data)
                test_loss, test_accuracy, out_dict = eval_loss_and_accuracy(model, test_inputs, test_labels, criterion)
                if config.log_to_wandb:
                    wandb.log({'test_loss': test_loss.item(), 'iter': n})
                    wandb.log({'test_accuracy': test_accuracy.item(), 'iter': n})
                    if config.save_weights:
                        fig1, ax1 = plt.subplots()
                        ax1.imshow(out_dict['block_0']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                        wandb.log({'l0_attn_map_test': fig1, 'iter': n})  # note: now we're logging the mean of the attention weights across data points
                        fig2, ax2 = plt.subplots()
                        ax2.imshow(out_dict['block_1']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                        wandb.log({'l1_attn_map_test': fig2, 'iter': n})
                        plt.close('all')

                # evaluate on ICLs
                icl_loss, icl_accuracy, out_dict = eval_loss_and_accuracy(model, test_inputs_ic, test_labels_ic, criterion)
                if config.log_to_wandb:
                    wandb.log({'icl_loss': icl_loss.item(), 'iter': n})
                    wandb.log({'icl_accuracy': icl_accuracy.item(), 'iter': n})
                    if config.save_weights:
                        fig1, ax1 = plt.subplots()
                        ax1.imshow(out_dict['block_0']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                        wandb.log({'l0_attn_map_icl': fig1, 'iter': n})  # note: now we're logging the mean of the attention weights across data points
                        fig2, ax2 = plt.subplots()
                        ax2.imshow(out_dict['block_1']['weights'].mean(axis=0).squeeze(), cmap=ATTENTION_CMAP)
                        wandb.log({'l1_attn_map_icl': fig2, 'iter': n})
                        plt.close('all')


                # evaluate on IWL
                iwl_loss, iwl_accuracy, out_dict = eval_loss_and_accuracy(model, test_inputs_iw, test_labels_iw, criterion)
                if config.log_to_wandb:
                    wandb.log({'iwl_loss': iwl_loss.item(), 'iter': n})
                    wandb.log({'iwl_accuracy': iwl_accuracy.item(), 'iter': n})

                print(f'iter {n}, loss: {loss}, ic_accuracy: {icl_accuracy}, iw_accuracy: {iwl_accuracy}')


if __name__ == '__main__':
    # possibly override config with command line arguments
    config = set_config(config)
    # run experiment
    main(config)
