import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utility import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Originally from https://github.com/sebastianruder/sluice-networks/blob/master/predictors.py
# Modified to work for MLPs
class CrossStitchLayer(nn.Module):
    """Cross-stitch layer class."""
    def __init__(self, model, num_tasks, layer_size, num_subspaces, init_scheme, i_layer):
        """
        Initializes a CrossStitchLayer.
        :param model: the parent NN model
        :param num_tasks: the number of tasks
        :param layer_size: the # of hidden dimensions of the previous LSTM layer
        :param num_subspaces: the number of subspaces
        :param init_scheme: the initialization scheme; "balanced" or "imbalanced"
        """

        super().__init__()

        assert int(layer_size / num_subspaces) == layer_size / num_subspaces, "ERROR: Layer size not divisible into subspaces"

        alpha_params = np.full((num_tasks * num_subspaces,
                                num_tasks * num_subspaces),
                               1. / (num_tasks * num_subspaces), dtype=np.float32)
        if init_scheme == "imbalanced":
            if num_subspaces == 1:
                alpha_params = np.full((num_tasks, num_tasks), 0.1 / (num_tasks - 1), dtype=np.float32)
                for i in range(num_tasks):
                    alpha_params[i, i] = 0.9
            else:
                alpha_params = np.array([[0.95,0.05],[0.05,0.95]], dtype=np.float32).repeat(num_tasks, axis=0).repeat(num_tasks, axis=1)

        self.alphas = nn.Parameter(torch.from_numpy(alpha_params).to(device))
        model.register_parameter("alpha_{}".format(i_layer), self.alphas)

        self.num_tasks = num_tasks.item()
        self.num_subspaces = num_subspaces
        self.layer_size = layer_size
        self.subspace_size = int(layer_size / num_subspaces)

    def stitch(self, x):

        # Split x into different subspaces.
        # Splits subspaces in the feature dimension and appends to the task dimension
        # This allows for easy transformation by multiplication with alpha tensor
        # [n_batch x d_hidden x n_task] => [n_batch x d_subspace x n_task.n_subspace]
        x = torch.cat(torch.split(x, self.subspace_size, dim=1), axis=2)

        # Ensure alphas add to one
        a = F.softmax(self.alphas, dim=0)

        # Perform stitch (linear combination of task/subspaces)
        # [n_batch x d_subspace x n_task.n_subspace][n_task.n_subspace x n_task.n_subspace]
        #       => [n_batch x d_subspace x n_task.n_subspace]
        x = torch.matmul(x, a)

        # Change dimension back, reversing first transformation
        # [n_batch x d_subspace x n_task.n_subspace] => [n_batch x d_hidden x n_task]
        x = torch.cat(torch.split(x, self.num_tasks, dim=2), axis=1)

        return x


class LayerStitchLayer(nn.Module):
    """Layer-stitch layer class."""
    def __init__(self, model, num_layers, init_scheme):
        """
        Initializes a LayerStitchLayer.
        :param model: the parent NN model
        :param num_layers: the number of layers in the network
        :param init_scheme: the initialisation scheme; balanced or imbalanced
        """

        super().__init__()

        # Remove one because sluice logic doesn't count output layer as a layer
        self.num_layers = num_layers - 1

        if init_scheme == "imbalanced":
            beta_params = np.full((self.num_layers), 0.1 / (self.num_layers - 1), dtype=np.float32)
            beta_params[-1] = 0.9
        elif init_scheme == "balanced":
            beta_params = np.full((self.num_layers), 1. / self.num_layers, dtype=np.float32)
        else:
            raise ValueError('Invalid initialization scheme for layer-stitch units: %s.' % init_scheme)
        self.betas = nn.Parameter(torch.from_numpy(beta_params).to(device))
        model.register_parameter("betas", self.betas)

    def stitch(self, xs):
        """
        Takes as input the predicted states of all the layers of a task-specific
        network and produces a linear combination of them.
        :param xs: a list of length num_layers containing network activations of
                   size [n_batch x d_hidden x n_tasks]
        :return: a linear combinations of the activations
        """

        # Ensure betas sum to one
        b = F.softmax(self.betas, dim=0)

        # Multiply activations by beta coefficients
        # [n_batch x d_hidden x n_task x n_layers][n_layers] -> [n_batch x d_hidden x n_task]
        xs = torch.stack(xs, axis=-1)
        combination = torch.matmul(xs, b)

        return combination


# Crosssttitch network for linear MLPs
# Acts as crossstitch network or sluice network depending on parameters
class MTL_Net_CS(nn.Module):

    def __init__(self, num_inputs, layer_sizes, num_tasks, use_layer_stitch, num_subspaces, alpha_init, beta_init=None, orthogonal_loss_coef=None):
        super(MTL_Net_CS, self).__init__()

        self.num_inputs = num_inputs
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_tasks = num_tasks
        self.use_cross_stitch = True
        self.use_layer_stitch = use_layer_stitch
        self.num_subspaces = num_subspaces
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.orthogonal_loss_coef = orthogonal_loss_coef

        prev_layer_size = num_inputs

        # Define shared Layers
        for i_layer in range(self.num_layers):

            layer_size = self.layer_sizes[i_layer]

            # Create tensors that define linear layer for each task
            lin = nn.Linear(prev_layer_size, layer_size)
            W = nn.Parameter(torch.stack([lin.weight]*num_tasks, axis=-1))
            b = nn.Parameter(torch.stack([lin.bias]*num_tasks, axis=-1))

            # Register tensors
            self.register_parameter("layer{}_W".format(i_layer), W)
            self.register_parameter("layer{}_b".format(i_layer), b)

            # Define layer that performs crossstitch operation between layers
            if self.use_cross_stitch and i_layer < self.num_layers-1:
                cross_stitch = CrossStitchLayer(self, self.num_tasks, layer_size, num_subspaces, alpha_init, i_layer)
                self.add_module("layer{}_cross_stitch".format(i_layer), cross_stitch)

            prev_layer_size = layer_size

        if self.use_layer_stitch:
            self.layer_stitch = LayerStitchLayer(self, self.num_layers, beta_init)
            self.add_module("layer_stitch", self.layer_stitch)


    def forward(self, x):

        # Duplicate for each task to allow parallel computations for all tasks
        x = torch.stack([x]*self.num_tasks, axis=-1)

        if self.use_layer_stitch:
            xs = []

        for i_layer in range(self.num_layers-1):

            # Perform linear layer. We use einsum to broadcast across task axis
            # b=batch index, i = input index, j = output index, t = task index
            W = getattr(self, "layer{}_W".format(i_layer))
            b = getattr(self, "layer{}_b".format(i_layer))
            x = torch.einsum("bit,jit->bjt", x, W) + b

            if self.use_cross_stitch:
                cs_layer = getattr(self, "layer{}_cross_stitch".format(i_layer))
                x = cs_layer.stitch(x)

            x = F.relu(x)

            if self.use_layer_stitch:
                xs.append(x)

        # Final layer
        if self.use_layer_stitch:
            x = self.layer_stitch.stitch(xs)

        W = getattr(self, "layer{}_W".format(self.num_layers-1))
        b = getattr(self, "layer{}_b".format(self.num_layers-1))
        x = torch.einsum("bit,jit->bjt", x, W) + b

        return x


    # Calculates the additinoal loss term in sluice networks that enforces the
    # two subspaces to be orthogonal.Only sensible when n_subspaces == 2
    def get_orthogonal_loss(self):

        assert self.num_subspaces == 2, "ERROR: Too many subspaces for orthogonal loss computation."

        loss = 0
        for i_layer in range(self.num_layers-1):

            W = getattr(self, "layer{}_W".format(i_layer))
            subspace_size = int(W.shape[0] / self.num_subspaces)
            W_subset1 = W[:subspace_size,:,:]
            W_subset2 = W[subspace_size:,:,:]

            loss = loss + torch.mean((W_subset1 - W_subset2)**2)

        loss = loss * self.orthogonal_loss_coef
        return loss
