import argparse
import json
import itertools
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

STR2OPTIM = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD}
TASK2PARAM = {              # Map from task name to param file.
    "text_cls": "params/params_text.json",
    "speech_cls": "params/params_speech.json",
    "images_cls": "params/params_images.json"
}


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
        "--cross_validation",
        help="For each model, find and display the best learning rate and optimal epoch using cross-validation.",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Print the loss during training. Should not be activated when testing for execution time.",
        action="store_true",
    )

    parser.add_argument(
        "--size_dataset_sample",
        help="If specified with an integer value, we can limit the size of the dataset in order to perform "
             "training faster.",
        default=None,
        type=int
    )

    parser.add_argument(
        "--train_test_split",
        help="float, the percentage of data to keep as test. Default 0.1",
        default=0.1,
        type=float
    )

    parser.add_argument(
        "--batch_size",
        help="int, the batch size for the train and test data loader.",
        default=64,
        type=int
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


def get_best_parameter(val_accuracies: np.array, best_param: object, best_cv_accuracy: float,
                       best_cv_epoch: int, param: object, optimizer: str, verbose: bool = False):
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
    df = pd.DataFrame(val_accuracies).melt(var_name='Epochs', value_name='Accuracy')
    accuracy_df = df.groupby('Epochs').agg({"Accuracy": ["count", "mean"]})
    # Discard epochs that have not been reached by all cross validation attempts.
    max_epochs_df = accuracy_df[  # count__max means that all attempts have reached such epoch
        accuracy_df[('Accuracy', 'count')] == accuracy_df[('Accuracy', 'count')].max()]
    best_accuracy_mean = max_epochs_df[('Accuracy', 'mean')].max()  # Get the best mean accuracy
    best_epoch = max_epochs_df[  # Get epoch which obtained a best accuracy mean
        max_epochs_df[('Accuracy', 'mean')] == best_accuracy_mean
        ].index.tolist()[-1]  # Select the largest epoch with best mean (there should be only one).
    if verbose:
        print("Best accuracy mean: {}, obtained at epoch {}".format(best_accuracy_mean, best_epoch))
    if best_param is None or best_cv_accuracy < best_accuracy_mean:
        # Update best parameters
        best_param = param
        best_cv_epoch = best_epoch
        best_cv_accuracy = best_accuracy_mean
        if verbose:
            print("update best param for {}:\nepochs = {}\naccuracy = {}\n params = {}".format(
                optimizer,
                best_cv_epoch,
                best_cv_accuracy,
                best_param
            ))
    else:
        if verbose:
            print("No improvements, best accuracy so far is {}".format(best_cv_accuracy))
    # Do some visualization stuff here!
    # sns.pointplot(x="Epochs", y="Accuracy",  kind='box', data=df)\
    #     .set_title("Validation accuracy during cross validation")
    # plt.show()
    return best_param, best_cv_epoch, best_cv_accuracy
