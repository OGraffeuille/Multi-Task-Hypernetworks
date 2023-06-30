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
from .tensor_op_MRN import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTL architecture
# Task specific layers are effectively fully connected layers,
# a mask matrix is used during feedforward calculations to prevent
# connections between nodes of different tasks
class MTL_Net(nn.Module):

    def __init__(self, num_inputs, shared_layer_sizes, task_layer_sizes, num_tasks, is_MRN, MRN_weight=None, MRN_feat_k=None):
        super(MTL_Net, self).__init__()

        self.num_inputs = num_inputs
        self.shared_layer_sizes = shared_layer_sizes
        self.task_layer_sizes = task_layer_sizes
        self.num_tasks = num_tasks
        self.num_outputs = self.task_layer_sizes[-1] if len(task_layer_sizes) > 0 else self.shared_layer_sizes[-1]
        self.is_MRN = is_MRN
        if self.is_MRN:
            self.MRN_feat_k = MRN_feat_k
            self.MRN_weight = MRN_weight

        prev_layer_size = num_inputs

        # Define shared Layers
        self.shared_layers = nn.ModuleList()
        for shared_layer_size in self.shared_layer_sizes:
            self.shared_layers.append(nn.Linear(prev_layer_size, shared_layer_size))
            prev_layer_size = shared_layer_size

        # Define task-specific Layers
        if len(self.task_layer_sizes) > 0:
            assert len(self.task_layer_sizes) < 2, "ERROR: only one task-specific layer implemented"
            task_layer_size = self.task_layer_sizes[0] #always 1 for regression
            self.task_layer = nn.Linear(prev_layer_size, task_layer_size * num_tasks)

            if self.is_MRN:
                self.task_layer.weight.data.normal_(0, 0.01)
                self.task_layer.bias.data.fill_(0.0)
                self.task_cov = nn.Parameter(torch.eye(self.num_tasks))
                self.class_cov = nn.Parameter(torch.eye(self.num_outputs))
                self.feature_cov = nn.Parameter(torch.eye(prev_layer_size))


    def forward(self, x):

        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)
            if (len(self.task_layer_sizes) > 0) or (i < len(self.shared_layers)-1):
                x = F.relu(x)

        if len(self.task_layer_sizes) > 0:
            x = self.task_layer(x)

        return x

    # Updates covariance matrices of MRN method
    def update_covs(self):

        def select_func(x):
            if x > 0.1:
                return 1. / x
            else:
                return x

        #fig, axs = plt.subplots(1,2,figsize=(10,5))
        #shw = axs[0].imshow(self.task_cov.data.detach().cpu().numpy())
        #plt.colorbar(shw)
        #shw = axs[1].imshow(self.feature_cov.data.detach().cpu().numpy())
        #plt.colorbar(shw)
        #plt.show()


        weights = self.task_layer.weight.data.view(self.num_tasks, self.task_layer_sizes[-1], self.shared_layer_sizes[-1]).contiguous() ############################

        # task covariance
        temp_task_cov = UpdateCov(weights.data, self.class_cov.data, self.feature_cov.data)
        u, s, v = torch.svd(temp_task_cov)
        s = s.cpu().apply_(select_func).cuda()
        temp_task_cov = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(temp_task_cov)
        if this_trace > 3000.0:
            self.task_cov = nn.Parameter(temp_task_cov / this_trace * 3000.0)
        else:
            self.task_cov = nn.Parameter(temp_task_cov)


        # feature covariance
        temp_feature_cov = UpdateCov(weights.data.permute(2, 0, 1).contiguous(), self.task_cov.data, self.class_cov.data)
        u, s, v = torch.svd(temp_feature_cov)
        s = s.cpu().apply_(select_func).cuda()
        temp_feature_cov = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
        this_trace = torch.trace(temp_feature_cov)

        k_max = 3000.
        if this_trace > k_max:
            #self.feature_cov = nn.Parameter(self.feature_cov + 1/self.MRN_feat_k * temp_feature_cov / this_trace * k_max)
            self.feature_cov = nn.Parameter(self.feature_cov + 1/self.MRN_feat_k * temp_feature_cov / this_trace * k_max)
        else:
            #self.feature_cov = nn.Parameter(self.feature_cov + 1/self.MRN_feat_k * temp_feature_cov)
            self.feature_cov = nn.Parameter(self.feature_cov + 1/self.MRN_feat_k * temp_feature_cov)


    # MRN Loss func. to encourage model weights to follow tensor normal distribution defined by cov matrices
    def get_cov_loss(self):
        assert self.is_MRN, "ERROR: Loss term only used for MRN architecture."
        weights = self.task_layer.weight.view(self.num_tasks, self.num_outputs, -1).contiguous()
        multi_task_loss = MultiTaskLoss(weights, self.task_cov, self.class_cov, self.feature_cov)
        return multi_task_loss[0] * self.MRN_weight
