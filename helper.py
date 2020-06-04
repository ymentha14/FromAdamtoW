import argparse
import json
import itertools
from copy import copy

import torch
import torch.optim as optim

STR2OPTIM = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD}


def parse_arguments():
    """Parses the global parameters from the command line arguments."""

    parser = argparse.ArgumentParser(description="From Adam to AdamW")

    parser.add_argument(
        "--task_name",
        default="all",
        help="By default execute all tasks. When specified, execute a specific task. "
        "Valid values: text_cls, speech_cls, imgages_cls.",
    )

    # (1): First phase, parameter optimization for each of the 3 datasets by a grid search approach.
    parser.add_argument(
        "--params_file",
        nargs="+",
        type=str,
        default="./params/params.json",
        help="Parameter file containing all different settings to conduct a GridSearch for the tasks. "
        "Provide 3 of them if you run all for the grid search, 1 otherwise.",
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
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device
