import pandas as pd
import numpy as np
import sys, time
from sklearn.utils import shuffle
import itertools
import copy

# Load generated datasets simply
#                  data_name - preset name for dataset being used
#               STL_metadata - whether to get STL w/ Metadata version of data (metadata joined with data)
#                 STL_onehot - whether to get STL w/ onehot version of data (no metadata, data has onehot encodings of task)
#                  test_frac - fraction of data to be used for test set (ignored if dataset already has train/test split)
#                  vali_frac - fraction of data to be used for validation set
#          vali_shuffle_seed - numpy Shuffle seed when separating train/vali
#          test_shuffle_seed - numpy Shuffle seed when separating test from train+vali
#                  num_tasks - only use the first n tasks
#         num_train_per_task - only use the first n instances in each task for training(+vali). Can also be  alist of length (num_task)
#       add_one_hot_features - replace int features with one-hot vectors
#             one_hot_labels - replace label with one-hot vector
def get_data(data_name, STL_metadata=False, STL_onehot=False,                                                          \
             test_frac=0.2, vali_frac=0.2, vali_shuffle_seed=0, test_shuffle_seed=0,                                   \
             num_tasks="all", num_train_per_task="all",                                                                \
             add_one_hot_features=False, one_hot_labels=False):
    filenames_dict = {
        "Cubic":                r"cubic\cubic_{}.csv",
        "Water Quality":        r"water_quality\water_quality_{}.csv",
        "Algorithms":           r"algorithms\algorithms_{}.csv",
        "Robot Arm":            r"robot_arm\robot_arm_{}.csv"
    }

    assert not (STL_metadata and STL_onehot), "ERROR: Can't use both STL and onehot"
    assert data_name in filenames_dict, "ERROR: Unknown data name: \"{}\". Data names are [{}].".format(data_name, ", ".join(key for key in filenames_dict))
    assert not (type(num_train_per_task) is list and len(num_train_per_task) != num_tasks), "ERROR: Wrong length of num_train_per_task when used as list."


    np.random.seed(test_shuffle_seed)

    output_data_dict = {}

    data_type = "data"
    if STL_metadata: data_type = "STL_metadata"
    if STL_onehot: data_type = "STL_onehot"
    data_filename = filenames_dict[data_name].format(data_type)
    data = pd.read_csv("data//" + data_filename)

    label_cols = [col for col in data if col.startswith("label")]
    X = data.drop(label_cols + ["task"], axis=1)
    y = data[label_cols]
    T = data[["task"]]
    if add_one_hot_features:
        X = _add_one_hot_features(X, T)
    if one_hot_labels:
        y = _one_hot_labels(y)

    # Reduce the number of tasks/instances in the dataset first
    if num_tasks != "all":
        X = X[T['task'] < num_tasks]
        y = y[T['task'] < num_tasks]
        T = T[T['task'] < num_tasks]
        if STL_onehot:
            X = X.drop([c for c in X.columns if c.startswith("t_") and int(c.split("_")[-1]) >= num_tasks], axis=1)

    # Separate into test, validation and training sets
    inds_train = []
    inds_vali = []
    inds_test = []
    for task in T["task"].unique():
        inds_task = T.index[T['task'] == task].tolist()
        n_tot = len(inds_task)

        # If not a special case, just use frac variables to determine number of test instances
        n_test = int(test_frac * n_tot)
        n_train = n_tot - n_test

        # If setting the number of training instances per task
        if num_train_per_task != "all":
            if type(num_train_per_task) is list:
                n = num_train_per_task.pop(np.random.randint(0,len(num_train_per_task)))
            else:
                n = num_train_per_task
            n_test = np.maximum(0, n_tot - n)
            n_train = n_tot - n_test

        # Separate test data from train(+vali) data
        inds_task_shuf = shuffle(inds_task, random_state=test_shuffle_seed) # Separate test by test seed first
        inds_test += inds_task_shuf[:n_test]
        inds_nottest = inds_task_shuf[n_test:n_test+n_train]

        # Separate train from vali data
        n_vali = int(n_train * vali_frac / (1 - test_frac))
        inds_nottest_shuf = shuffle(inds_nottest, random_state=vali_shuffle_seed)
        inds_vali += inds_nottest_shuf[:n_vali]
        inds_train += inds_nottest_shuf[n_vali:]

    # Return correct indices
    output_data_dict["X_train"] = X.values[inds_train]
    output_data_dict["y_train"] = y.values[inds_train]
    output_data_dict["T_train"] = T.values[inds_train]
    output_data_dict["X_vali"] = X.values[inds_vali]
    output_data_dict["y_vali"] = y.values[inds_vali]
    output_data_dict["T_vali"] = T.values[inds_vali]
    output_data_dict["X_test"] = X.values[inds_test]
    output_data_dict["y_test"] = y.values[inds_test]
    output_data_dict["T_test"] = T.values[inds_test]

    # Metadata
    if not (STL_metadata or STL_onehot):
        metadata_filename = filenames_dict[data_name].format("metadata")
        metadata = pd.read_csv("data//" + metadata_filename)
        if num_tasks != "all":
            metadata = metadata[metadata["_task"] < num_tasks]
        output_data_dict["metadata"] = metadata

    return output_data_dict

# Add one-hot features
def _add_one_hot_features(X, T):
    for t in T["task"].unique():
        X["is_{}".format(t)] = (T.values == t).astype(int)
    return X

# Replace int labels by one-hot labels
def _one_hot_labels(y):
    y = pd.DataFrame(y)
    for unique_class in y["label"].unique():
        y["label_{}".format(int(unique_class))] = (y["label"].values == unique_class)
    y = y.drop("label", axis=1)
    return y



# Takes a dictionary of parameters.
# List parameters are treated as a list of parameter values to iterate through.
# Creates an iterator that goes through every parameter combination.
# Useful for experiments.
class param_iterator():

    def __init__(self, *params):

        # Array of parameter dicts
        self.params = [*params]

        self.iter_param_keys = []
        self.iter_param_vals = []

        self.num_combinations = 1

        # For each param in each params dict given,
        for params in self.params:
            for key in list(params):

                # if the param is in the form of a list then iterate through it
                if isinstance(params[key], list):
                    if len(params[key]) == 1:
                        params[key] = params[key][0]
                    elif len(params[key]) > 1:
                        self.iter_param_keys.append(key)
                        self.iter_param_vals.append(params[key])
                        self.num_combinations *= len(params[key])


        # Rotate through seed last, so that if we stop experiments early, we have some seed runs for all params
        if "seed" in self.iter_param_keys:
            seed_val = self.iter_param_vals[self.iter_param_keys.index("seed")]
            self.iter_param_keys.remove("seed")
            self.iter_param_keys.insert(0, "seed")
            self.iter_param_vals.remove(seed_val)
            self.iter_param_vals.insert(0, seed_val)

        # Check no two parameters have same name (would cause errors)
        assert len(self.iter_param_keys) == len(set(self.iter_param_keys)), "ERROR: Multiple iterative parameters with same name"

        # Generate all permutations of iterable parameters
        self.param_combinations = itertools.product(*self.iter_param_vals)

        # Prepare to iterate through params
        self.t0 = time.time()
        self.param_combinations_ind = 0

        print("There are {} parameter combinations to iterate through.".format(self.num_combinations))
        for i in range(len(self.iter_param_keys)):
            if len(self.iter_param_vals[i]) > 1:
                print(" - {} - {}".format(self.iter_param_keys[i], self.iter_param_vals[i]))


    # Returns the next parameter combination
    def next(self):

        self.param_combinations_ind += 1
        t = time.time() - self.t0
        print(" ## NEW PARAMETER COMBINATION {}/{} ({:.0f}s)".format(self.param_combinations_ind, self.num_combinations, t))

        params_to_return = copy.deepcopy(self.params)
        iter_param_vals = next(self.param_combinations)
        for i in range(len(iter_param_vals)):
            key = self.iter_param_keys[i]
            val = iter_param_vals[i]
            for p in params_to_return:
                if key in p:
                    p[key] = val

            print(" ##  {:20} = {}".format(key, val))

        print(params_to_return)
        return (params for params in params_to_return)
