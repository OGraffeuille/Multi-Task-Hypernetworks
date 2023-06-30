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

import matplotlib
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# MTL algorithms
from . import MTL, TF, CS

# Utility functions
from ..utility import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generic class that can train any of the baseline MTL architectures
# depending on input parameters.
class train_MTL():

    def __init__(self, params, data_params):

        # Copy all parameters
        for key in params:
            setattr(self, key, params[key])

        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Do all the things
        self.prepare_data(data_params)

        self.initialise_model()

        self.optimise_model()

        self.evaluate_model()

        if self.plot:
            self.plot_results()


    def prepare_data(self, p):

        data = get_data(**p, vali_shuffle_seed=self.seed, test_shuffle_seed=self.seed)

        for key in data:
            setattr(self, key+"_df", data[key])

        self.num_inputs = self.X_train_df.shape[1]
        self.num_tasks = np.concatenate([self.T_train_df, self.T_vali_df, self.T_test_df]).max()+1
        #self.num_tasks = len(np.unique(np.concatenate([self.T_train_df, self.T_vali_df, self.T_test_df])))
        self.num_train = len(self.T_train_df)

        # Convert to tensors
        self.X_train = torch.from_numpy(self.X_train_df).float().to(device)
        self.X_vali = torch.from_numpy(self.X_vali_df).float().to(device)
        self.X_test = torch.from_numpy(self.X_test_df).float().to(device)
        self.y_train = torch.from_numpy(self.y_train_df).float().to(device)
        self.y_vali = torch.from_numpy(self.y_vali_df).float().to(device)
        self.y_test = torch.from_numpy(self.y_test_df).float().to(device)
        self.T_train = torch.from_numpy(self.T_train_df).long().to(device)
        self.T_vali = torch.from_numpy(self.T_vali_df).long().to(device)
        self.T_test = torch.from_numpy(self.T_test_df).long().to(device)

        # Task
        self.has_metadata = hasattr(self, "metadata_df")
        if self.has_metadata:
            self.T_names_unique = self.metadata_df["_task_name"].values
        self.T_unique = torch.arange(self.num_tasks)
        self.T_count = torch.histc(torch.cat([self.T_train, self.T_vali]), len(self.T_unique))

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
            self.y_vali = torch.from_numpy(self.y_vali_df).float().to(device)#.view(-1,1)
            self.y_mu = torch.mean(torch.cat([self.y_train, self.y_vali], axis=0), axis=0)
            self.y_std = torch.std(torch.cat([self.y_train, self.y_vali], axis=0), axis=0)
            self.y_train = torch.div(torch.add(self.y_train, -self.y_mu), self.y_std)
            self.y_vali = torch.div(torch.add(self.y_vali, -self.y_mu), self.y_std)
            self.y_test = torch.div(torch.add(self.y_test, -self.y_mu), self.y_std)

        self.dataset_train = torch.utils.data.TensorDataset(self.X_train, self.y_train, self.T_train)
        self.dataset_vali = torch.utils.data.TensorDataset(self.X_vali, self.y_vali, self.T_vali)
        self.dataset_test = torch.utils.data.TensorDataset(self.X_test, self.y_test, self.T_test)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.batch_shuffle)


    # Initialise the model architecture and other relevant variables
    def initialise_model(self):

        # Default params
        p_default = {
                "vali_epoch_freq": 5,
                "vali_epoch_delay": 20,
                "max_epochs": 2000,
                "batch_size": 64,
                "batch_shuffle": True,
                "is_MRN": False,
                "is_TF": False,
                "is_CS": False,
                "is_sluice": False
            }
        for key in p_default:
            if not hasattr(self, key):
                setattr(self, key, p_default[key])

        # Metrics
        self.metrics = {}
        self.metrics["seed"] = self.seed
        self.metrics["lr"] = self.lr
        self.metrics["batch_size"] = self.batch_size
        self.metrics["batch_shuffle"] = self.batch_shuffle
        self.metrics["num_inputs"] = self.num_inputs
        self.shared_layer_sizes = self.arch[0]
        self.task_layer_sizes = self.arch[1]
        self.metrics["shared_layers"] = ",".join([str(size) for size in self.shared_layer_sizes])
        self.metrics["task_layers"] = ",".join([str(size) for size in self.task_layer_sizes])
        self.metrics["vali_epoch_freq"] = self.vali_epoch_freq
        self.metrics["vali_epoch_delay"] = self.vali_epoch_delay
        if self.record_task_sizes:
            self.metrics["task_sizes"] = [t.item() for t in self.T_count]

        # MRN-specific params
        self.metrics["is_MRN"] = self.is_MRN
        if self.is_MRN:
            self.metrics["MRN_weight"] = self.MRN_weight
            self.metrics["MRN_feat_k"] = self.MRN_feat_k

        # Tensor factorisation specific params
        self.metrics["is_TF"] = self.is_TF
        if self.is_TF:
            self.metrics["TF_method"] = self.TF_method
            self.metrics["TF_k"] = self.TF_k

        # Crossstitch & sluice specific params
        self.metrics["is_CS"] = self.is_CS
        self.metrics["is_sluice"] = self.is_sluice
        if self.is_sluice:
            self.metrics["sluice_num_subspaces"] = self.sluice_num_subspaces
            self.metrics["sluice_alpha_init"] = self.sluice_alpha_init
            self.metrics["sluice_beta_init"] = self.sluice_beta_init
            self.metrics["sluice_orthogonal_loss_coef"] = self.sluice_orthogonal_loss_coef

        self.is_MTL = len(self.task_layer_sizes) > 0

        # Check variables
        assert (self.is_TF + self.is_MRN + self.is_CS + self.is_sluice) < 2, "ERROR: Multiple model types selected."
        assert not (self.is_TF and len(self.shared_layer_sizes) > 0), "ERROR: Can't be both shared layers with factorisation model."

        # Create models
        if self.is_TF:
            self.model = TF.MTL_Net_TF(self.num_inputs, self.task_layer_sizes, self.num_tasks, self.TF_method, self.TF_k)
        elif self.is_CS:
            self.model = CS.MTL_Net_CS(self.num_inputs, self.task_layer_sizes, self.num_tasks, False, 1, "balanced")
        elif self.is_sluice:
            self.model = CS.MTL_Net_CS(self.num_inputs, self.task_layer_sizes, self.num_tasks, True, self.sluice_num_subspaces,
                                       self.sluice_alpha_init, self.sluice_beta_init, self.sluice_orthogonal_loss_coef)
        elif self.is_MRN:
            self.model = MTL.MTL_Net(self.num_inputs, self.shared_layer_sizes, self.task_layer_sizes, self.num_tasks, True, self.MRN_weight, self.MRN_feat_k)
        else:
            self.model = MTL.MTL_Net(self.num_inputs, self.shared_layer_sizes, self.task_layer_sizes, self.num_tasks, False)

        parameter_dict = [{"params": self.model.parameters()}]
        self.optimizer = optim.Adam(parameter_dict, lr=self.lr)

        if self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
        elif self.loss_func_name == "ce":
            self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            assert False, "ERROR: Unknown loss func name"

        assert not (self.is_MRN and len(self.task_layer_sizes) == 0), "ERROR: MRN requires MTL architecture."


    # Optimise a traditional multi-task learning model
    def optimise_model(self):

        print("MTL - Learning rate: {:.6f}, seed: {}, {} features".format(self.lr, self.seed, self.num_inputs))

        self.loss_train_hist = []
        self.loss_vali_hist = []

        epoch = 0
        while epoch < self.max_epochs:

            # Training data
            train_loss_epoch = 0
            for data in self.train_loader:
                X_local, y_local, T_local = data

                self.optimizer.zero_grad()
                loss = self._run_model(X_local, y_local, T_local, train_mode=True) * len(y_local)
                loss.backward()
                self.optimizer.step()

                train_loss_epoch += loss.detach().cpu().numpy()

            loss_train = train_loss_epoch / self.num_train
            self.loss_train_hist.append(loss_train)

            # MRN: update covariance matrices at end of each epoch
            if self.is_MRN:
                self.model.update_covs()

            # Validation
            if np.mod(epoch, self.vali_epoch_freq) == 0:

                loss_vali = self._run_model(self.X_vali, self.y_vali, self.T_vali)
                self.loss_vali_hist.append(loss_vali)

                if self.verbose:
                    print("LR: {:.6f}, Epoch: {:4d}/{:4d}, Train | Validation Loss: {:.4f} | {:.4f}".format(self.lr, epoch, self.max_epochs, loss_train, loss_vali))

                delay = math.ceil(self.vali_epoch_delay / self.vali_epoch_freq)
                if len(self.loss_vali_hist) > delay and self.loss_vali_hist[-1] >= self.loss_vali_hist[-1-delay]:
                    break

            epoch += 1

        self.metrics["num_epochs"] = epoch


    def evaluate_model(self):

        acc_train, task_acc_train, self.pred_train = self._run_model(self.X_train.float(), self.y_train, self.T_train, eval_mode=True, compute_task_losses=True)
        acc_vali, task_acc_vali, self.pred_vali = self._run_model(self.X_vali.float(), self.y_vali, self.T_vali, eval_mode=True, compute_task_losses=True)
        acc_test, task_acc_test, self.pred_test = self._run_model(self.X_test.float(), self.y_test, self.T_test, eval_mode=True, compute_task_losses=True)

        self.metrics[self.loss_func_name + "_train"] = acc_train.item()
        self.metrics[self.loss_func_name + "_vali"] = acc_vali.item()
        self.metrics[self.loss_func_name + "_test"] = acc_test.item()
        for task in self.T_unique:
            task = str(task.item())
            self.metrics[self.loss_func_name + "_train_" + task] = task_acc_train[task].item()
            self.metrics[self.loss_func_name + "_vali_" + task] = task_acc_vali[task].item()
            self.metrics[self.loss_func_name + "_test_" + task] = task_acc_test[task].item()

        print("{} features, LR: {}, seed: {}, epochs: {}, acc: {:.6f}".format(self.num_inputs, self.lr, self.seed, self.metrics["num_epochs"], acc_test))


    def plot_results(self):

        # Learning curve plot
        fig, axs = plt.subplots(1, 1, figsize=(16, 5))
        axs.plot(self.loss_train_hist, label="Train lr={:.5f}".format(self.lr))#, c="C"+str(seed_i), linestyle=":")
        axs.plot(np.arange(len(self.loss_vali_hist))*self.vali_epoch_freq, self.loss_vali_hist, label="Vali")#, c="C"+str(seed_i))

        axs.set(xlabel="Epochs", ylabel="Loss", title="Loss curve")
        axs.legend()
        plt.show()

        # Prediction goals
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))

        # Plotting variables
        cMap = plt.get_cmap('hsv')
        cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=len(self.T_unique))
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cMap)

        dataset_names = ["train", "validation", "test"]
        X_vars = [self.X_train[:,0], self.X_vali[:,0], self.X_test[:,0]]
        y_vars = [self.y_train, self.y_vali, self.y_test]
        pred_vars = [self.pred_train, self.pred_vali, self.pred_test]
        T_vars = [self.T_train, self.T_vali, self.T_test]
        for lst in [X_vars, y_vars, pred_vars, T_vars]:
            for ind in range(len(lst)):
                lst[ind] = lst[ind].flatten().cpu().numpy()

        min_val = np.min(np.concatenate(y_vars + pred_vars))
        max_val = np.max(np.concatenate(y_vars + pred_vars))

        for i in range(3):

            # Plot x=y line
            axs.flat[i].plot([min_val, max_val], [min_val, max_val], ":", alpha=0.5)

            # Scatters
            dataset_name = dataset_names[i]
            y_var = y_vars[i] * self.y_std.item() + self.y_mu.item()
            pred_var = pred_vars[i]
            T_var = T_vars[i]

            for j in self.T_unique:
                task = j.item()
                task_label = self.T_names_unique[task] if self.has_metadata else task
                axs[i].scatter(y_var[T_var == task], pred_var[T_var == task], s=15, alpha=0.5, color=scalarMap.to_rgba(task), label=task_label)

            var_name = "ln(Chl [mg/m3])"
            axs.flat[i].set(xlabel="Real "+var_name, ylabel="Predicted "+var_name, title=dataset_name+" dataset")
        plt.show()

        # Predict
        if True:
            fig, axs = plt.subplots(1, 3, figsize=(16, 5))
            for i in range(3):
                y_var = y_vars[i] * self.y_std.item() + self.y_mu.item()
                axs.flat[i].scatter(X_vars[i], y_var, label="True")
                axs.flat[i].scatter(X_vars[i], pred_vars[i], label="Pred")
                axs.flat[i].legend()
            plt.show()


    def _run_model(self, X, y, T, train_mode=False, eval_mode=False, compute_task_losses=False):

        # Change network mode
        if not train_mode:
            torch.set_grad_enabled(False)
            self.model.eval()

        # Compute Prediction for all tasks
        pred = self.model(X)

        # Only keep relevant task predictions
        if self.is_MTL:
            pred = pred.view(len(T), -1, self.num_tasks)
            pred = pred[torch.arange(len(T)), :, T.flatten()]

        # Un-normalise regression target if required
        if eval_mode and not self.is_classification:
            pred = pred * self.y_std + self.y_mu
            y = y * self.y_std + self.y_mu

        # Compute loss
        loss = self.loss_func(pred, y)

        # Multiply losses such that tasks with fewer instances get greater loss
        if self.normalise_task_training:
            mult = 1 / self.T_count[T]
            loss = loss * mult / torch.mean(mult)

        # Find the per-task average loss
        if compute_task_losses:
            task_losses = {str(task.item()) : torch.mean(loss[T.flatten() == task]) for task in self.T_unique}

        loss = torch.mean(loss)

        # Compute additional training losses if necessary
        if train_mode:
            if self.is_sluice:
                loss = loss + self.model.get_orthogonal_loss()
            if self.is_MRN:
                loss = loss + self.model.get_cov_loss()

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
