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

from scipy.linalg.interpolative import svd

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Implementation of "Deep Multi-Task Representation Learning: A Tensor
# "Factorisation Approach" for linear NNs. Somewhat based on example code from
# https://github.com/wOOL/DMTRL/blob/master/demo.ipynb
# model params  factorisation_method: method of factorising weight tensors of
#               linear layers from different tasks. "Tucker" or "TT" are implemented
#               factorisation_k: parameter for factorisation (roughly, the
#               dimension  compress the weight tensors to)
class MTL_Net_TF(nn.Module):

    def __init__(self, num_inputs, layer_sizes, num_tasks, method, k):
        super(MTL_Net_TF, self).__init__()

        assert (method == "Tucker") or (method == "TT"), "Only Tucker and TT methods are implemented."

        self.num_inputs = num_inputs
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_tasks = num_tasks
        self.method = method
        self.k = k

        prev_layer_size = num_inputs

        for i_layer in range(self.num_layers):

            layer_size = layer_sizes[i_layer]

            # Create tensors that define linear layer for each task
            lin = nn.Linear(prev_layer_size, layer_size)
            W = lin.weight.detach().cpu().numpy()
            b = lin.bias.detach().cpu().numpy()
            W = np.stack([W]*num_tasks, axis=-1)
            b = np.stack([b]*num_tasks, axis=-1)

            # Factorise the weight tensors
            # Take the min() because we can't factorise to a smaller dimension than current tensor size
            k = min(self.k, prev_layer_size, layer_size)
            _, W_factors = TensorProducer(W, self.method, eps_or_k=k, return_true_var=True)

            # Save model parameters
            for factor_name in W_factors:
                if isinstance(W_factors[factor_name], list):
                    for i_factor in range(len(W_factors[factor_name])):
                        arr = W_factors[factor_name][i_factor]
                        tensor = nn.Parameter(torch.from_numpy(arr).float().to(device))
                        self.register_parameter("layer{}_Wfactor{}{}".format(i_layer, factor_name, i_factor), tensor)
                else:
                    arr = W_factors[factor_name]
                    tensor = nn.Parameter(torch.from_numpy(arr).float().to(device))
                    self.register_parameter("layer{}_Wfactor{}".format(i_layer, factor_name), tensor)
            tensor = nn.Parameter(torch.from_numpy(b).float().to(device))
            self.register_parameter("layer{}_b".format(i_layer), tensor)

            prev_layer_size = layer_size

    def forward(self, x):

        # Duplicate for each task to allow parallel computations for all tasks
        x = torch.stack([x]*self.num_tasks, axis=-1)

        for i_layer in range(self.num_layers):

            if self.method == "Tucker":

                W = getattr(self, "layer{}_WfactorS".format(i_layer))
                for i_factor in range(3):
                    U = getattr(self, "layer{}_WfactorU{}".format(i_layer, i_factor))
                    W = torch.tensordot(W, U, [(0,), (1,)])
                b = getattr(self, "layer{}_b".format(i_layer))

                # Perform linear layer. We use einsum to broadcast across task axis
                # b=batch index, i = input index, j = output index, t = task index
                x = torch.einsum("bit,jit->bjt", x, W) + b
                if i_layer < self.num_layers-1:
                    x = F.relu(x)

            elif self.method == "TT":

                U0 = getattr(self, "layer{}_WfactorU0".format(i_layer))
                U1 = getattr(self, "layer{}_WfactorU1".format(i_layer))
                U2 = getattr(self, "layer{}_WfactorU2".format(i_layer))
                W = torch.tensordot(U0, U1, [(-1,),(0,)])
                W = torch.tensordot(W, U2, [(-1,),(0,)])
                W = W.squeeze(0).squeeze(-1)
                b = getattr(self, "layer{}_b".format(i_layer))

                # Perform linear layer. We use einsum to broadcast across task axis
                # b=batch index, i = input index, j = output index, t = task index
                x = torch.einsum("bit,jit->bjt", x, W,) + b
                if i_layer < self.num_layers-1:
                    x = F.relu(x)

        return x



def my_svd(A, eps_or_k=0.01):
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    U, S, V = svd(A, eps_or_k, rand=False)

    return U, S, V.T


def t_unfold(A, k):
    A = np.transpose(A, np.hstack([k, np.delete(np.arange(A.ndim), k)]))
    A = np.reshape(A, [A.shape[0], np.prod(A.shape[1:])])

    return A


def t_dot(A, B, axes=(-1, 0)):
    return np.tensordot(A, B, axes)


def tt_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = [min(np.prod(n[:i + 1]), np.prod(n[i + 1:])) for i in range(d - 1)]

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * (d - 1)

    r = [1] * (d + 1)

    TT = []
    C = A.copy()

    for k in range(d - 1):
        C = C.reshape((r[k] * n[k], int(C.size / (r[k] * n[k]))))
        (U, S, V) = my_svd(C, eps_or_k[k])
        r[k + 1] = U.shape[1]
        TT.append(U[:, :r[k + 1]].reshape((r[k], n[k], r[k + 1])))
        C = np.dot(np.diag(S[:r[k + 1]]), V[:r[k + 1], :])
    TT.append(C.reshape(r[k + 1], n[k + 1], 1))

    return TT


def tucker_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = list(n)

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * d

    U = [my_svd(t_unfold(A, k), eps_or_k[k])[0] for k in range(d)]
    S = A
    for i in range(d):
        S = t_dot(S, U[i], (0, 0))

    return U, S


def tt_cnst(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = t_dot(S, A[i + 1])

    return np.squeeze(S, axis=(0, -1))


def tucker_cnst(U, S):
    for i in range(len(U)):
        S = t_dot(S, U[i], (0, 1))

    return S


def TensorUnfold(A, k):
    tmp_arr = np.arange(A.ndim)
    A = np.transpose(A, [tmp_arr[k]] + np.delete(tmp_arr, k).tolist())
    shapeA = list(A.shape)
    A = np.reshape(A, [shapeA[0], np.prod(shapeA[1:])])

    return A


def TensorProduct(A, B, axes=(-1, 0)):
    shapeA = list(A.shape)
    shapeB = list(B.shape)
    shapeR = np.delete(shapeA, axes[0]).tolist() + np.delete(shapeB, axes[1]).tolist()
    result = np.matmul(np.transpose(TensorUnfold(A, axes[0])), TensorUnfold(B, axes[1]))

    return np.reshape(result, shapeR)


def TTTensorProducer(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = TensorProduct(S, A[i + 1])

    return S.squeeze(0).squeeze(-1)


def TuckerTensorProducer(U, S):
    for i in range(len(U)):
        S = TensorProduct(S, U[i], (0, 1))

    return S


def TensorProducer(X, method, eps_or_k=0.01, datatype=np.float32, return_true_var=False):
    if method == 'Tucker':
        U, S = tucker_dcmp(X, eps_or_k)
        #U = [tf.Variable(i.astype(datatype)) for i in U]
        U = [i for i in U]
        #S = tf.Variable(S.astype(datatype))
        W = TuckerTensorProducer(U, S)
        param_dict = {'U': U, 'S': S}
    elif method == 'TT':
        A = tt_dcmp(X, eps_or_k)
        A = [i for i in A]
        W = TTTensorProducer(A)
        param_dict = {'U': A}
    elif method == 'LAF':
        U, S, V = my_svd(np.transpose(t_unfold(X, -1)), eps_or_k)
        #U = tf.Variable(U.astype(datatype))
        #V = tf.Variable(np.dot(np.diag(S), V).astype(datatype))
        V = np.dot(np.diag(S), V)
        #W = tf.reshape(tf.matmul(U, V), X.shape)
        W = np.reshape(np.matmul(U, V), X.shape)
        param_dict = {'U': U, 'V': V}
    if return_true_var:
        return W, param_dict
    else:
        return W
