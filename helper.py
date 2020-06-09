import argparse
import json
import itertools
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import KFold

from pathlib import Path

from datetime import datetime

str_2_optimizer = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD}


def get_param_filepath(task_name: str, best: bool):
    """
    Given a task_name, return it's params filename
    """

    if best:

        best_param = {  # Map from task name to param file.
            "text_cls": "params/best_text.json",
            "speech_cls": "params/best_speech.json",
            "images_cls": "params/best_images.json",
        }

        return best_param[task_name]

    else:

        grid_search_param = {  # Map from task name to param file.
            "text_cls": "params/grid_search_text.json",
            "speech_cls": "params/grid_search_speech.json",
            "images_cls": "params/grid_search_images.json",
        }

        return grid_search_param[task_name]


def get_log_filepath(task_name: str):
    """
    ...
    """

    TASK2LOGFILE = {  # Map from task name to param file.
        "text_cls": "log/log_text_results.json",
        "speech_cls": "log/log_speech_results.json",
        "images_cls": "log/log_images_results.json",
    }

    return TASK2LOGFILE[task_name]


def get_default_num_epochs(task_name: str):
    # TODO: fine tune it for the specific task! Write Done when you did and remove the todo!
    task_2_numepochs = {  # Map from task name to param file.
        "text_cls": 10,
        "speech_cls": 20,
        "images_cls": 150,  # Done
    }

    return task_2_numepochs[task_name]


def parse_arguments():
    """Parses the global parameters from the command line arguments."""

    parser = argparse.ArgumentParser(description="From Adam to AdamW")

    parser.add_argument(
        "--task_name",
        default="all",
        help="By default execute all tasks. When specified, execute a specific task. "
        "Valid values: text_cls, speech_cls, images_cls.",
    )

    # (2): Second phase: more executions (in order to obtain a robust estimate) and longer ones.
    parser.add_argument(
        "--optimizer",
        default="all",
        help="By default experiment with all optimizers. When specified, execute a specific optimizer. "
        "Valid values: 'Adam', 'AdamW', 'SGD' or 'all'. Incompatible with parameter param_file.",
    )

    parser.add_argument(
        "--num_runs",
        help=f"Number of independent execution runs for the robust estimate. Default 5.",
        default=5,
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
        "--patience",
        help="Patience to use for early stopping. Default is 7.",
        default=7,
        type=int,
    )

    parser.add_argument(
        "--k",
        help="k value used during cross validation. Default is 3.",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--overwrite_best_param",
        help="If we are performing grid_search and we specify this parameter, the parameter in best_'task'.json will "
        "be updated with the best parameters found with the current grid search.",
        action="store_true",
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


def compute_best_parameter(
    val_accuracies: np.array,
    best_param: object,
    best_cv_epoch: int,
    best_cv_accuracy: float,
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
         best_cv_epoch: epoch the best model has achieved the best accuracy so far.
         best_cv_accuracy: best accuracy of the best model so far.
         param: hyper parameters of the current model (which may become best_param)
         optimizer: optimizer used
         verbose: define True to print more information.
    """
    # This builds a 2 columns dataframe, one column with epoch, the other with accuracy
    df = pd.DataFrame(val_accuracies).melt(var_name="Epochs", value_name="Accuracy")
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

    return best_param, best_cv_epoch, best_cv_accuracy


def log(
    log_filepath: str,
    task_name: str,
    train_losses: list,
    train_accuracies: list,
    val_losses: list,
    val_accuracies: list,
    test_losses: list,
    test_accuracies: list,
    optimizer: str,
    param: object,
    num_epochs: int,
):
    """append the scores of the current run to the json in log_path

    Log the cross validation results (train and validation accuracy and losses, as a 2-dimensional list divided by attempt and epoch). Set the cross_validation flag to true.

    Args:
        log_filepath: the path for the log file (specific for every task)
        task_name: str, name of the task (images_cls, text_cls, speech_cls)
        train_losses: 2-dimensional list indexed by attempt (k-fold) and epoch
        train_accuracies: 2-dimensional list indexed by attempt (k-fold) and epoch
        val_losses: 2-dimensional list indexed by attempt (k-fold) and epoch
        val_accuracies: 2-dimensional list indexed by attempt (k-fold) and epoch
        test_losses: 2-dimensional list indexed by attempt (k-fold) and epoch
        test_accuracies: 2-dimensional list indexed by attempt (k-fold) and epoch
        optimizer: name of the optimizer (Adam, AdamW, SGD).
        param: object with hyperparameters for the optimizer tested (lr, betas, momentum etc)
        num_epochs: int, number of epochs the model has been run when training using best params.
            It is meaningful only when test_accuracies and test_losses are not none.
    """

    log_path_posix = Path(log_filepath)
    if not log_path_posix.exists():
        print("Creating log file")
        with open(log_filepath, "w") as f:
            json.dump([], f, indent=4)

    date = datetime.now().strftime("%m_%d_%y-%H_%M_%S")

    new_record = {}

    new_record["date"] = date
    new_record["task_name"] = task_name
    new_record["train_losses"] = train_losses
    new_record["train_accuracies"] = train_accuracies

    new_record["val_losses"] = val_losses
    new_record["val_accuracies"] = val_accuracies

    new_record["test_losses"] = test_losses
    new_record["test_accuracies"] = test_accuracies

    new_record["optimizer"] = str(optimizer)
    new_record["param"] = str(param)
    new_record["num_epochs"] = num_epochs

    with open(log_filepath, "r") as f:
        old_log = json.load(f)

    old_log.append(new_record)

    with open(log_filepath, "w") as f:
        json.dump(old_log, f, indent=4)


def get_best_parameters(task_name):
    """
    Return JSON containing the best parameters.
    """

    filepath = get_param_filepath(task_name, best=True)
    with open(filepath, "r") as f:
        return json.load(f)


def override_best_parameters(task_name: str, best_param: object):
    """
    Override the best parameters for the task given
    """

    filepath = get_param_filepath(task_name, best=True)
    with open(filepath, "w") as f:
        json.dump(best_param, f, indent=4)
