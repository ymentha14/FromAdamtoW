import argparse
import json
import itertools
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import KFold

STR2OPTIM = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD}


TASK2LOGFILE = {  # Map from task name to param file.
    "text_cls": "log/log_text_results.json",
    "speech_cls": "log/log_speech_results.json",
    "images_cls": "log/log_images_results.json",
}


def get_param_filename(task_name):
    """
    Given a task_name, return it's params filename
    """

    TASK2PARAM = {  # Map from task name to param file.
        "text_cls": "params/params_text.json",
        "speech_cls": "params/params_speech.json",
        "images_cls": "params/params_images.json",
    }

    return TASK2PARAM[task_name]


def parse_arguments():
    """Parses the global parameters from the command line arguments."""

    parser = argparse.ArgumentParser(description="From Adam to AdamW")

    parser.add_argument(
        "--task_name",
        default="all",
        help="By default execute all tasks. When specified, execute a specific task. "
        "Valid values: text_cls, speech_cls, images_cls.",
    )

    parser.add_argument(
        "--max_cross_validation_epochs",
        help=f"Maximal number of epochs to use in cross-validation training. Default 100.",
        default=100,
        type=int,
    )

    # (2): Second phase: more executions (in order to obtain a robust estimate) and longer ones.
    parser.add_argument(
        "--optimizer",
        default="all",
        help="By default experiment with all optimizers. When specified, execute a specific optimizer. "
        "Valid values: Adam, AdamW, SGD or 'all'. Incompatible with parameter param_file.",
    )

    parser.add_argument(
        "--learning_rate",
        help="Learning rate. If not specified, the model will use the best learning "
        "rate found during cross-validation.",
        default=None,
        type=float,
    )

    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train. If not specified, "
        "the model will use the best number found during cross-validation.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--num_runs",
        help=f"Number of independent execution runs for the robust estimate. Default 10.",
        default=10,
    )

    parser.add_argument(
        "--grid_search",
        help="For each task, find the best parameters for each optimizer.",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Print the loss during training. Should not be activated when testing for execution time.",
        action="store_true",
    )

    parser.add_argument(
        "--sample_size",
        help="Limit the number of examples during training.",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--train_size_ratio",
        help="Percentage of total data to use for training. Default is 0.8",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        help="int, the batch size for the train and test data loader. Default is 32.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--seed",
        help="Random seed used to reproduce the same exact results.",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--patience", help="Patience to use for early stopping", default=7, type=int,
    )

    return parser.parse_args()


def get_params_combinations(path_to_json):
    """
    create all the parameter combinations possible wrt to the values present in the json.
    
    Args:
        path_to_json (str): relative path to json file for cross validation
        
    Returns:
        Dictionnary of all possible combinations for each optimizer with the form:
        >>> {'Adam':[
                        {'lr':0.01,'b1':0.04,'b2':0.03},
                        {'lr':0.01,'b1':0.04,'b2':0.04},
                        ...
                    ],
             'SGD': [
                        ...
                    ],
             ...
             }
    """
    with open(path_to_json) as f:
        json_object = json.load(f)

    # returned dic
    comb_tot = {}
    for optim, param in json_object.items():
        # keys describes the parameter names for this optimizer
        keys = param.keys()
        # values = list of list of values for the parameters
        values = param.values()
        # create all possible combinations as a list
        combinations = list(itertools.product(*values))
        # recreate the dictionary structure for each parameter
        dict_combinations = [
            {key: value for key, value in zip(keys, comb)} for comb in combinations
        ]
        comb_tot[optim] = dict_combinations
    return comb_tot


def adapt_params(params):
    """
    adapts the name of all params such that they match the ones from the pytorch
    
    Args:
        params: dict of params following our nomenclature
        
    Returns:
        corr_params: same dict with correct keys to match kwargs for pytorch
    """
    corr_params = copy(params)
    if "beta1" in params or "beta2" in params:
        assert "beta1" in params and "beta2" in params
        corr_params["betas"] = (params["beta1"], params["beta2"])
        del corr_params["beta1"], corr_params["beta2"]
    return corr_params


def get_device():
    """
    Get the device, CUDA or CPU depending on the machine availability.
    Returns:
        device: CUDA or CPU
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def get_best_parameter(
    val_accuracies: np.array,
    best_param: object,
    best_cv_accuracy: float,
    best_cv_epoch: int,
    param: object,
    optimizer: str,
    verbose: bool = False,
):
    """
    Given a 2-dimensional list of accuracies per epoch (first dimension: k-th attempt,
    second dimension: epoch), return the best epoch, mean accuracy (mean computed
    over the same epoch) and param among the current best (best_cv_accuracy, best_cv_epoch, param)
    and the one computed over val_accuracies (which stores the validation accuracies computed in the cv attempt).
    Args:
         val_accuracies: 2-dimensional array, stores validation accuracies computed in the last cv attempt.
         best_param: parameters of model which has best accuracy so far.
         best_cv_accuracy: best accuracy of the best model so far.
         best_cv_epoch: epoch the best model has achieved the best accuracy so far.
         param: hyper parameters of the current model (which may become best_param)
         optimizer: optimizer used
         verbose: define True to print more information.
    """
    # This builds a 2 columns dataframe, one column with epoch, the other with accuracy
    df = pd.DataFrame(val_accuracies.tolist()).melt(
        var_name="Epochs", value_name="Accuracy"
    )
    accuracy_df = df.groupby("Epochs").agg({"Accuracy": ["count", "mean"]})
    # Discard epochs that have not been reached by all cross validation attempts.
    max_epochs_df = accuracy_df[  # count__max means that all attempts have reached such epoch
        accuracy_df[("Accuracy", "count")] == accuracy_df[("Accuracy", "count")].max()
    ]
    best_accuracy_mean = max_epochs_df[
        ("Accuracy", "mean")
    ].max()  # Get the best mean accuracy
    best_epoch = max_epochs_df[  # Get epoch which obtained a best accuracy mean
        max_epochs_df[("Accuracy", "mean")] == best_accuracy_mean
    ].index.tolist()[
        -1
    ]  # Select the largest epoch with best mean (there should be only one).
    if verbose:
        print(
            "Best accuracy mean: {}, obtained at epoch {}".format(
                best_accuracy_mean, best_epoch
            )
        )
    if best_param is None or best_cv_accuracy < best_accuracy_mean:
        # Update best parameters
        best_param = param
        best_cv_epoch = best_epoch
        best_cv_accuracy = best_accuracy_mean
        if verbose:
            print(
                "update best param for {}:\nepochs = {}\naccuracy = {}\n params = {}".format(
                    optimizer, best_cv_epoch, best_cv_accuracy, best_param
                )
            )
    else:
        if verbose:
            print(
                "No improvements, best accuracy so far is {}".format(best_cv_accuracy)
            )
    # Do some visualization stuff here!
    # sns.pointplot(x="Epochs", y="Accuracy",  kind='box', data=df)\
    #     .set_title("Validation accuracy during cross validation")
    # plt.show()
    return best_param, best_cv_epoch, best_cv_accuracy


def log_results(
    results: object,
    best_cv_epoch: int,
    best_param: object,
    optimizer: str,
    log_file: str,
):
    """
    Log the results to the specified file.
    Object is a Python object, so it can be saved directly as a json.
    log are going to be appended at the end of the log file, in order to avoid losing data.
    data, the json taken by the file, must be a list (so it should be initialized as an empty list).
    Args:
        results: json with results (accuracy and loss per each parameter)
        best_cv_epoch: int, the best epoch to train the model-optimizer combination
        best_param: the best parameter found for the optimizer
        optimizer: the optimizer used
        log_file: path to the log file
    """
    try:
        with open(log_file, "r") as json_file:
            data = json.loads(json_file.read())
        # print(data)
        if not data:
            data = []
    except:  # Should everything bad happen, just re-initialize it with an empty list.
        print("Something went wrong with the log file!")
        data = []
    data.append(
        {
            "results": results,
            "best_cv_epoch": best_cv_epoch,
            "best_param": best_param,
            "optimizer": optimizer,
            "cross_validation": False,  # This is the best result, not one of the many cross validation attempts
        }
    )
    with open(log_file, "w") as json_file:
        json.dump(data, json_file)


def log_results_cross_validation(
    train_losses: list,
    train_accuracies: list,
    val_losses: list,
    val_accuracies: list,
    optimizer: str,
    log_file: str,
):
    """
    Log the cross validation results (train and validation accuracy and losses, as a 2-dimensional list
    divided by attempt and epoch). Set the cross_validation flag to true.
    Args:
        train_losses: 2-dimensional list indexed by attempt (k-fold) and epoch
        train_accuracies: 2-dimensional list indexed by attempt (k-fold) and epoch
        val_losses: 2-dimensional list indexed by attempt (k-fold) and epoch
        val_accuracies: 2-dimensional list indexed by attempt (k-fold) and epoch
        optimizer: name of the optimizer (Adam, AdamW, SGD).
        log_file: name of the log file
    """
    with open(log_file, "r") as json_file:
        data = json.loads(json_file.read())
    if not data:
        data = []
    data.append(
        {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "optimizer": optimizer,
            "cross_validation": True,  # This is just one of the many cross validation results.
        }
    )
    with open(log_file, "w") as json_file:
        json.dump(data, json_file)


def split_train_test(dataset, train_ratio):
    """
    Split dataset into two parts
    """

    full_dataset = _get_full_dataset()

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def split_kfold(dataset, k, batch_size):
    """
    Split the dataset with k-fold

    Return an iterable (generator) where each element is a tuple (train_dataloader, val_dataloader) with batch_size batch_size
    """

    kfold = KFold(n_splits=k)

    for train_index, val_index in kfold.split(dataset):

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        yield (train_dataloader, val_dataloader)


def split_k_times(dataset, k, train_ratio, batch_size):
    """
    Split the dataset k-times with a given train ratio

    Return an iterable (generator)
    """

    for _ in range(k):
        train_dataset, val_dataset = split_train_test(dataset, train_ratio)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )

        yield (train_dataloader, val_dataloader)
