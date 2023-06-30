import os
import sys
import warnings

import math
import random
import numpy as np
import glob
import datetime
import time
import datetime
import copy
import json

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utility import *

class HyperNet(nn.Module):

    def __init__(self, target_arch, num_tasks, embedding_dim, hidden_dim, embeddings_share, init_embedding_var, init_embedding_same,
            hyper_extractor_layers, metadata=None, shuf_metadata=False):
        super(HyperNet, self).__init__()

        self.target_arch = target_arch
        self.num_tasks = num_tasks
        self.num_layers = len(target_arch)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_metadata = metadata is not None
        self.embeddings_share = embeddings_share
        self.init_embedding_same = init_embedding_same
        self.init_embedding_var = init_embedding_var
        self.hyper_extractor_layers = hyper_extractor_layers

        # Initialise task embeddings
        # We initialise embeddings from uniform distribution [-a, a] where a = sqrt(3 * var)
        # where var is the desired variance of the embeddings
        if embedding_dim > 0:
            a = np.sqrt(3 * self.init_embedding_var)
            num_unique_embeddings = 1 if self.embeddings_share else num_tasks
            if self.init_embedding_same:
                task_embeddings = torch.Tensor(1, embedding_dim).uniform_(-a, a).repeat(num_unique_embeddings, 1)
            else:
                task_embeddings = torch.Tensor(num_unique_embeddings, embedding_dim).uniform_(-a, a)
            self.task_embeddings = nn.Parameter(task_embeddings)
        else:
            self.task_embeddings = None

        # Initialise metadata
        if self.use_metadata:
            self.task_metadata = torch.div(torch.add(metadata, -torch.mean(metadata, axis=0)), torch.std(metadata, axis=0))
            assert torch.sum(torch.isnan(self.task_metadata)) == 0, "ERROR: Invalid metadata, maybe one column is all zeros?"

            if shuf_metadata: # sensitivity / debug test to see if metadata helps
                self.task_metadata = self.task_metadata[torch.randperm(self.task_metadata.shape[0])]

            self.task_metadata = nn.Parameter(self.task_metadata)

        self.metadata_dim = self.task_metadata.shape[1] if self.use_metadata else 0

        # Shared hyper extractor
        if self.hyper_extractor_layers > 0:
            self._hyper_extractor = nn.Sequential()
            for i in range(self.hyper_extractor_layers):
                d_in = self.hidden_dim if i > 0 else self.embedding_dim + self.metadata_dim
                d_out = self.hidden_dim
                self._hyper_extractor.add_module("hyper_extractor_linear_{}".format(i), nn.Linear(d_in, d_out))
                self._hyper_extractor.add_module("hyper_extractor_relu_{}".format(i), nn.ReLU())
            self.add_module("hyper_extractor", self._hyper_extractor)

        # Implement hypernetwork into target_arch input dict
        i = 0
        for layer in self.target_arch:

            if layer["type"] == "linear_nohyper":
                i += 1
                layer["x_layer"] = nn.Linear(layer["params"][0], layer["params"][1])
                self.add_module("{}_layer".format(i), layer["x_layer"])

            elif layer["type"] == "conv_nohyper":
                i += 1
                layer["x_layer"] = nn.Conv2d(layer["params"][0], layer["params"][1], layer["params"][2])
                self.add_module("{}_layer".format(i), layer["x_layer"])

            elif layer["type"] == "linear":

                layer["index"] = i
                i += 1

                d_in = layer["params"][0]
                d_out = layer["params"][1]
                d_h = self.hidden_dim if self.hyper_extractor_layers > 0 else self.embedding_dim + self.metadata_dim
                n_t = self.num_tasks

                # To generate weight matrix for x
                # Different hypernetwork weight initialisations were tested, but
                # had equal or worse performance than this naive initialisaiton
                layer["_xweight_generator_weight1"] = nn.Parameter(data=nn.Linear(d_h, d_in).weight.t()[:,:,None,None])         # [d_h x d_in x 1 x 1]
                layer["_xweight_generator_weight2"] = nn.Parameter(data=nn.Linear(d_in, d_out).weight.t()[None,:,:,None])       # [1 x d_in x d_out x 1]
                layer["_xweight_generator_weight3"] = nn.Parameter(data=nn.Linear(d_h, d_out).weight.t()[:,None,:,None])        # [d_h x 1 x d_out x 1]
                layer["_xweight_generator_bias"] = nn.Parameter(data=nn.Linear(d_in, d_in*d_out).bias.reshape(d_in, d_out))    # [d_in x d_out]]

                # To generate bias vector for x
                layer["_xbias_generator_layer"] = nn.Linear(d_h, d_out)

                # Add parameters to model
                self.register_parameter("{}_xweight_generator_weight1".format(i), layer["_xweight_generator_weight1"])
                self.register_parameter("{}_xweight_generator_weight2".format(i), layer["_xweight_generator_weight2"])
                self.register_parameter("{}_xweight_generator_weight3".format(i), layer["_xweight_generator_weight3"])
                self.register_parameter("{}_xweight_generator_bias".format(i), layer["_xweight_generator_bias"])
                self.add_module("{}_xbias_generator_layer".format(i), layer["_xbias_generator_layer"])

    def forward(self, x):

        for layer in self.target_arch:

            # HyperNet variables part
            if "linear" in layer["type"]:
                h = self.task_embeddings
                if h is None:
                    assert self.use_metadata, "ERROR: If no embeddings, hypernetwork requires metadata input."
                    h = self.task_metadata
                else:
                    if self.embeddings_share:
                        h = h.repeat(self.num_tasks, 1)                         # If same embeddings for all tasks
                    if self.use_metadata:
                        h = torch.cat([h, self.task_metadata], axis=1)
                if self.hyper_extractor_layers > 0:
                    h = self._hyper_extractor(h)                                # [n_t x d_h + d_m -> n_t x d_h]

            if layer["type"] == "linear":

                # Generate weight tensor that will be used as funcion from h to xweight
                # This tensor is factorised to require fewer parameters
                xweight_generator_weight = layer["_xweight_generator_weight1"] * layer["_xweight_generator_weight2"] * layer["_xweight_generator_weight3"]

                # Generate weight tensor for x forward pass
                xweight = torch.matmul(
                        h[:,None,None,None,:],                                  # [d_t x d_h] -> [d_t x 1 x 1 x 1 x d_h]
                        xweight_generator_weight.permute(1,2,0,3)               #  -> [d_in x d_out x d_h x 1]
                    ).squeeze(dim=-1).squeeze(dim=-1) \
                    + layer["_xweight_generator_bias"]# [d_t x 1 x 1 x 1 x d_h][d_in x d_out x d_h x 1] -> [d_t x d_in x d_out x 1 x 1] -> [d_t x d_in x d_out]

                # Generate bias tensor for x
                xbias = layer["_xbias_generator_layer"](h)                      # [n_t x d_h -> n_t x d_out]

                # Pass x using generated weights
                x = torch.matmul(
                        x[:,:,None,:],                                          # [d_batch x n_t x d_in] -> [d_batch x n_t x 1 x d_in] (unsqueeze)
                        xweight
                    ).squeeze(dim=2) + \
                    xbias                                                       # [d_batch x n_t x 1 x d_in][n_t x d_in x d_out] -> [d_batch x n_t x 1 x d_out] -> [d_batch x n_t x d_out]
                

            elif layer["type"] == "conv_nohyper":
                x = layer["x_layer"](x)

            elif layer["type"] == "max_pool":
                x = F.max_pool2d(x, kernel_size=layer["params"][0], stride=layer["params"][1])
            elif layer["type"] == "flatten":
                x = x.view(x.shape[0],-1)
            elif layer["type"] == "relu":
                x = F.relu(x)
            else:
                assert False, "Invalid layer name"

        return x



class train_HyperNet():

    def __init__(self, model_params, data_params):

        isvalid_params = self.prepare_params(model_params)
        if not isvalid_params:
            return None

        self.prepare_data(data_params)
        self.initialise_model()
        self.optimise_model()
        self.evaluate_model()


    # Process model paramaters from arguments
    #       p - model params
    def prepare_params(self, p):

        # Copy all parameters
        p = copy.deepcopy(p)

        self.metrics = {}
        for key in p:
            assert not hasattr(self, key), "ERROR: Duplicate parameter name."
            setattr(self, key, p[key])
            self.metrics[key] = p[key]
        self.metrics["target_arch"] = json.dumps(self.target_arch)

        # Default params
        p_default = {
                "vali_epoch_freq": 5,
                "vali_epoch_delay": 20,
                "max_epochs": 2000,
                "batch_size": 64,
                "normalise_task_training": False,
                "shuf_metadata": False,
                "embeddings_share": False,
                "hyper_extractor_layers": 0,
                "embedding_lr": 1,
                "metadata_lr": 0,
                "init_embedding_var": 1,
                "init_embedding_same": True,
                "get_vali_tasks_loss": True,
                "record_task_sizes": False,
                "is_classification": False,
                "loss_func_name": "mse"
            }
        for key in p_default:
            if not hasattr(self, key):
                setattr(self, key, p_default[key])
                self.metrics[key] = p_default[key]

        # Check params are valid
        if not self.use_metadata:
            if self.shuf_metadata:
                print("SKIPPING PARAMETER COMBINATION - shuffling metadata while having no metadata")
                return False

        # Torch defaults
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        return True

    # Process data into pytorch objects
    #       p - data params
    def prepare_data(self, p):

        # Copy all parameters
        p = copy.deepcopy(p)
        for key in p:
            assert not hasattr(self, key), "ERROR: Duplicate parameter name."
            setattr(self, key, p[key])

        # Load data
        data = get_data(**p, vali_shuffle_seed=self.seed, test_shuffle_seed=self.seed)

        for key in data:
            setattr(self, key+"_raw", data[key])
        del data

        self.num_inputs = self.X_train_raw.shape[1:]#len(self.X_train_df.columns)
        self.num_ouputs = self.y_train_raw.shape[1:]#len(self.X_train_df.columns)
        self.num_train = len(self.T_train_raw)
        self.num_tasks = np.concatenate([self.T_train_raw, self.T_vali_raw, self.T_test_raw]).max()+1
        #self.num_tasks = len(np.unique(np.concatenate([self.T_train_raw, self.T_vali_raw, self.T_test_raw])))
        if self.is_classification:
            self.num_classes = len(np.unique(np.concatenate([self.y_train_raw, self.y_vali_raw, self.y_test_raw])))

        # Convert to tensors
        self.X_train = torch.from_numpy(self.X_train_raw).float().to(self.device)
        self.X_vali = torch.from_numpy(self.X_vali_raw).float().to(self.device)
        self.X_test = torch.from_numpy(self.X_test_raw).float().to(self.device)
        self.y_train = torch.from_numpy(self.y_train_raw).float().to(self.device)
        self.y_vali = torch.from_numpy(self.y_vali_raw).float().to(self.device)
        self.y_test = torch.from_numpy(self.y_test_raw).float().to(self.device)
        self.T_train = torch.from_numpy(self.T_train_raw).long().to(self.device)
        self.T_vali = torch.from_numpy(self.T_vali_raw).long().to(self.device)
        self.T_test = torch.from_numpy(self.T_test_raw).long().to(self.device)

        # Metadata
        # note: has_metadata is whether metadata exists (for example to use task_name)
        #       use_metadata is whether to use metadata in the hypernetwork
        self.T_names = torch.arange(self.num_tasks)
        self.has_metadata = False
        self.metadata = None
        if hasattr(self, "metadata_raw"):
            self.T_names = self.metadata_raw["_task_name"].values
            drop_cols = [col for col in self.metadata_raw.columns if col.startswith("_")]
            metadata_vals = self.metadata_raw.drop(drop_cols, axis=1).values
            if metadata_vals.size > 0:
                self.has_metadata = True
                if self.use_metadata:
                    self.metadata = torch.from_numpy(metadata_vals).float().to(self.device)

        # Task
        self.T_unique = torch.arange(self.num_tasks)
        self.T_count = torch.histc(torch.cat([self.T_train, self.T_vali]), len(self.T_unique))
        if self.record_task_sizes: #DELETE
            self.metrics["task_sizes"] = [t.item() for t in self.T_count]

        # X
        self.X_mu = torch.mean(torch.cat([self.X_train, self.X_vali], axis=0), axis=0)
        self.X_std = torch.std(torch.cat([self.X_train, self.X_vali], axis=0), axis=0)
        self.X_std[self.X_std == 0] = 1
        self.X_train = torch.div(torch.add(self.X_train, -self.X_mu), self.X_std)
        self.X_vali = torch.div(torch.add(self.X_vali, -self.X_mu), self.X_std)
        self.X_test = torch.div(torch.add(self.X_test, -self.X_mu), self.X_std)

        # y
        if self.is_classification:
            # Turn one-hot target into integers
            self.y_train = self.y_train.long()
            self.y_vali = self.y_vali.long()
            self.y_test = self.y_test.long()
        else:
            # Normalise regression target
            self.y_mu = torch.mean(torch.cat([self.y_train, self.y_vali], axis=0), axis=0)
            self.y_std = torch.std(torch.cat([self.y_train, self.y_vali], axis=0), axis=0)
            self.y_train = torch.div(torch.add(self.y_train, -self.y_mu), self.y_std)
            self.y_vali = torch.div(torch.add(self.y_vali, -self.y_mu), self.y_std)
            self.y_test = torch.div(torch.add(self.y_test, -self.y_mu), self.y_std)


        # Dataset
        self.dataset_train = torch.utils.data.TensorDataset(self.X_train, self.y_train, self.T_train)
        self.dataset_vali = torch.utils.data.TensorDataset(self.X_vali, self.y_vali, self.T_vali)
        self.dataset_test = torch.utils.data.TensorDataset(self.X_test, self.y_test, self.T_test)

        # Data Loader
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)#, sampler=sampler)

    # Initialise the model architecture and other relevant variables
    def initialise_model(self):

        # Model
        self.model = HyperNet(self.target_arch, self.num_tasks, self.embedding_dim, self.hidden_dim,
                              self.embeddings_share, self.init_embedding_var, self.init_embedding_same,
                              self.hyper_extractor_layers,
                              self.metadata, self.shuf_metadata).to(self.device)

        # Optimiser
        parameter_dict = []
        for p_name, p in self.model.named_parameters():
            if p_name == "task_embeddings":
                parameter_dict.append({"params": p, "lr":self.lr * self.embedding_lr})
            elif p_name == "task_metadata":
                parameter_dict.append({"params": p, "lr":self.lr * self.metadata_lr})
            else:
                parameter_dict.append({"params": p, "lr":self.lr})

        self.optimizer = optim.Adam(parameter_dict)

        if self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
        elif self.loss_func_name == "ce":
            self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            assert False, "ERROR: Unknown loss func name"

    # Optimise a traditional multi-task learning model
    def optimise_model(self):

        print("MTL HyperNetwork - Learning rate: {:.6f}, seed: {}, {} tasks, {} features".format(self.lr, self.seed, self.num_tasks, self.num_inputs))

        self.loss_train_hist = []
        self.loss_vali_hist = []
        self.loss_vali_tasks_hist = []

        self.epoch = 0
        while self.epoch < self.max_epochs:

            # Training data
            train_loss_epoch = 0
            for data in self.train_loader:

                X_batch, y_batch, T_batch = data

                self.optimizer.zero_grad()
                loss = self._run_model(X_batch, y_batch, T_batch, train_mode=True) * len(y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss_epoch += loss.detach().cpu().numpy()

            loss_train = train_loss_epoch / self.num_train
            #wandb.log({"loss_train": loss_train, "epoch": self.epoch})
            self.loss_train_hist.append(loss_train)

            # Validation
            if np.mod(self.epoch, self.vali_epoch_freq) == 0:

                #plt.imshow(self.model.task_embeddings.detach().cpu().numpy())
                #plt.colorbar()
                #plt.show()

                loss_vali = self._run_model(self.X_vali, self.y_vali, self.T_vali)
                self.loss_vali_hist.append(loss_vali)

                if self.get_vali_tasks_loss:
                    _, vali_task_acc, _ = self._run_model(self.X_vali, self.y_vali, self.T_vali, eval_mode=True, compute_task_losses=True)
                    loss_vali_tasks = [vali_task_acc[str(task.item())] for task in self.T_unique]
                    self.loss_vali_tasks_hist.append(loss_vali_tasks)
                    #wandb.log({"loss_vali_{:03d}".format(task): loss_tasks[task] for task in self.T_unique})

                if self.verbose:
                    print("LR: {:.6f}, Epoch: {:4d}/{:4d}, Train | Validation Loss: {:.4f} | {:.4f}".format(self.lr, self.epoch, self.max_epochs, loss_train, loss_vali))

                delay = math.ceil(self.vali_epoch_delay / self.vali_epoch_freq)
                if len(self.loss_vali_hist) > delay and self.loss_vali_hist[-1] >= self.loss_vali_hist[-1-delay]:
                    break

            self.epoch += 1

        self.metrics["num_epochs"] = self.epoch

    # Runs data through model
    def evaluate_model(self):

        acc_train, task_acc_train, self.pred_train = self._run_model(self.X_train, self.y_train, self.T_train, eval_mode=True, compute_task_losses=True)
        acc_vali, task_acc_vali, self.pred_vali = self._run_model(self.X_vali, self.y_vali, self.T_vali, eval_mode=True, compute_task_losses=True)
        acc_test, task_acc_test, self.pred_test = self._run_model(self.X_test, self.y_test, self.T_test, eval_mode=True, compute_task_losses=True)

        self.metrics["acc_train"] = acc_train.item()
        self.metrics["acc_vali"] = acc_vali.item()
        self.metrics["acc_test"] = acc_test.item()
        for task in self.T_unique:
            task = str(task.item())
            if not np.isnan(task_acc_train[task]): self.metrics["acc_train_" + task] = task_acc_train[task]
            if not np.isnan(task_acc_vali[task]): self.metrics["acc_vali_" + task] = task_acc_vali[task]
            if not np.isnan(task_acc_test[task]): self.metrics["acc_test_" + task] = task_acc_test[task]

        print("{} features, LR: {}, seed: {}, epochs: {}, acc: {:.4}|{:.4f}|{:.4f}".format(self.num_inputs, self.lr, self.seed, self.metrics["num_epochs"], acc_train, acc_vali, acc_test))


    # Runs the hypernetwork model with given data
    #              X, y, T - data
    #           train_mode - whether to set gradients to be learnt
    #            eval_mode - whether to return different metrics (e.g. accuracy over crossentropy)
    #  compute_task_losses - additionally return task-wise performance
    #    max_compute_batch - perform computation in batches reduce memory requirements (avoid doing whole test set in 1 batch)
    def _run_model(self, X, y, T, train_mode=False, eval_mode=False, compute_task_losses=False, max_compute_batch=1024):

        # Change network mode
        if not train_mode:
            torch.set_grad_enabled(False)
            self.model.eval()

        # Compute Prediction for all tasks. Perform in batches for memory efficiency
        X = X.unsqueeze(1).repeat(1, self.num_tasks, 1)
        X = [X[i*max_compute_batch : min(X.shape[0], (i+1)*max_compute_batch)] for i in range(int(X.shape[0]/max_compute_batch)+1)]
        pred = torch.cat([self.model(x) for x in X])

        # Extract correct task performance
        pred = pred[torch.arange(len(T)), T.flatten(), :]

        # Compute loss
        loss_func = self.loss_func

        if eval_mode:
            if self.is_classification:
                # Compute accuracy if required
                pred = torch.max(pred, dim=1)[1]
                def acc(true, pred):
                    return (true.flatten() == pred.flatten()) * 1.
                loss_func = acc
            else:
                # Un-normalise regression target if required
                pred = pred * self.y_std + self.y_mu
                y = y * self.y_std + self.y_mu

        loss = loss_func(pred, y)

        # Multiply losses such that tasks with fewer instances get greater loss
        if self.normalise_task_training:
            mult = 1 / self.T_count[T]
            loss = loss * mult / torch.mean(mult)

        # Find the per-task average loss
        if compute_task_losses:
            task_losses = {str(task.item()) : torch.mean(loss[T.flatten() == task]).item() for task in self.T_unique}

        loss = torch.mean(loss)

        # Change network mode if necessary
        if not train_mode:
            torch.set_grad_enabled(True)
            self.model.train()

        if eval_mode:
            if compute_task_losses:
                return loss, task_losses, pred
            else:
                return loss, pred
        else:
            return loss
