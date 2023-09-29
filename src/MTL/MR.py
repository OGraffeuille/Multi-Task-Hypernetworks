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


# Maximum Roaming network for linear MLPs
class MTL_Net_MR(nn.Module):

    def __init__(self, num_inputs, shared_layer_sizes, task_layer_sizes, num_tasks, p, update_freq):
        super(MTL_Net_MR, self).__init__()

        self.num_inputs = num_inputs
        self.shared_layer_sizes = shared_layer_sizes
        self.task_layer_sizes = task_layer_sizes
        self.num_tasks = num_tasks
        self.p = p
        self.update_freq = update_freq


        # Record which network parameters have been masked
        self.mask_history = []

        prev_layer_size = num_inputs

        # Define shared Layers
        for i_layer in range(len(self.shared_layer_sizes)):

            layer_capacity = self.shared_layer_sizes[i_layer]

            # Adjust layer_size to have redundant connections, such that task-wise masks still have layer_size capacity
            layer_size = math.ceil(layer_capacity / (1 - self.p))

            # Create tensors that define linear layer, shared between each task
            lin = nn.Linear(prev_layer_size, layer_size)
            W = nn.Parameter(lin.weight)
            b = nn.Parameter(lin.bias)

            # Define Maximum Roaming mask, different for each task
            mask = torch.zeros(layer_size, num_tasks)
            for t in range(num_tasks):
                mask[torch.randperm(layer_size)[:layer_capacity], t] = 1  
            
            # Record which mask elements have been activated
            self.mask_history.append(mask.data.detach().cpu().numpy())

            # Register tensors
            self.register_parameter("layer{}_W".format(i_layer), W)
            self.register_parameter("layer{}_b".format(i_layer), b)
            self.register_buffer("layer{}_mask".format(i_layer), mask)
            
            prev_layer_size = layer_size

        # Define task-specific heads Layer
        lin = nn.Linear(prev_layer_size, task_layer_sizes[0])
        W = nn.Parameter(torch.stack([lin.weight]*num_tasks, axis=-1))
        b = nn.Parameter(torch.stack([lin.bias]*num_tasks, axis=-1))

        # Register tensors
        self.register_parameter("task_layer_W", W)
        self.register_parameter("task_layer_b", b)
             
    

    def forward(self, x):

        # Duplicate for each task to allow parallel computations for all tasks
        x = torch.stack([x]*self.num_tasks, axis=-1)

        for i_layer in range(len(self.shared_layer_sizes)):

            # Perform linear layer. We use einsum to broadcast across task axis
            # b=batch index, i = input index, j = output index, t = task index
            # W and b are shared between all tasks, but each task uses a different mask.
            W = getattr(self, "layer{}_W".format(i_layer))
            b = getattr(self, "layer{}_b".format(i_layer))
            b = torch.stack([b]*self.num_tasks, axis=-1)

            x = torch.einsum("bit,ji->bjt", x, W) + b
            
            # Apply mask and ReLU, except last (regression) layer
            mask = getattr(self, "layer{}_mask".format(i_layer))
            x = torch.einsum("bjt,jt->bjt", x, mask)
                        
            x = F.relu(x)

        # Perform task-specific heads Layer
        W = getattr(self, "task_layer_W")
        b = getattr(self, "task_layer_b")
        x = torch.einsum("bit,jit->bjt", x, W) + b

        return x

    # Update Maximum Roaming masks
    def update_masks(self):
        
        # Check that all mask elements have not been activated
        first_layer_size = math.ceil(self.shared_layer_sizes[0] / (1 - self.p))
        
        if np.sum(self.mask_history[0]) < self.num_tasks * first_layer_size:
                
            for i_layer in range(len(self.shared_layer_sizes)):
                
                for t in range(self.num_tasks):

                    mask = getattr(self, "layer{}_mask".format(i_layer))[:,t].data.detach().cpu().numpy()
                    mask_history = self.mask_history[i_layer][:,t].flatten()

                    # Find values to activate and discard
                    to_activate = random.sample(np.where(1-mask_history)[0].tolist(), k=1)
                    to_discard = random.sample(np.where(mask)[0].tolist(), k=1)

                    # Update mask
                    getattr(self, "layer{}_mask".format(i_layer))[to_activate,t] = 1
                    getattr(self, "layer{}_mask".format(i_layer))[to_discard,t] = 0

                    # Update tested units
                    self.mask_history[i_layer][to_activate,t] = 1

