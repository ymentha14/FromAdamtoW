#!/usr/bin/env python3

"""
MAIN
"""
import math

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

import helper
from pathlib import Path
import numpy as np

from tasks import images_cls
from tasks import speech_cls
from tasks import text_cls
from tester import Tester


def grid_search(task, args):
    """
    Compute grid search for the given task
    """
    (task_name, task_model, train_dataset, X, scoring_func) = task

    if args.verbose:
        print("=" * 60 + f"\nGrid Search for tasks : {task_name}")

    param_filename = helper.get_param_filepath(task_name, best=False)

    print(param_filename)

    combinations = helper.get_params_combinations(param_filename)

    # start of the grid search
    if args.verbose:
        print(
            "Testing {} combinations in total".format(
                sum([len(i) for i in combinations.values()])
            )
        )
        print(f"Performing grid search on {len(train_dataset)} examples.")

    for optim, params in combinations.items():

        best_param = None
        best_cv_accuracy = None
        best_cv_epoch = None

        for i, param in enumerate(params):
            if args.verbose:
                print(f"\n{i}. Testing {optim} with {param}")
            tester = Tester(
                args=args,
                task_name=task_name,
                train_dataset=train_dataset,
                test_dataset=None,
                task_model=task_model,
                optimizer=optim,
                param=param,
                scoring_func=scoring_func,
                num_epochs=helper.get_default_num_epochs(task_name),
            )

            # Run the cross validation phase
            (train_losses, train_accuracies, val_losses, val_accuracies,) = tester.run(
                do_cv=True
            )

            # Update the best parameter combination, if the accuracy for this cross validation phase is higher
            (best_param, best_cv_epoch, best_cv_accuracy,) = helper.get_best_parameter(
                val_accuracies,
                best_param,
                best_cv_accuracy,
                best_cv_epoch,
                param,
                optim,
                True,
            )

        print("TODO. We need to store the information regarding the best parameters!")


def main():
    # setting-up logs
    LOG_DIRECTORY = Path("log")
    LOG_DIRECTORY.mkdir(exist_ok=True)
    LOG_FILENAME = LOG_DIRECTORY / "results.log"

    """logging.basicConfig(
        filename=LOG_FILENAME,
        filemode="a",
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b %H:%M",
        level=logging.INFO,
    )
    """

    # Get arguments
    args = helper.parse_arguments()

    # Set the torch seed
    torch.manual_seed(args.seed)

    tasks_to_evaluate = []

    if args.task_name == "images_cls" or args.task_name == "all":
        task = (
            "images_cls",
            images_cls.get_model(),
            *images_cls.get_train_test_dataset(
                args.seed, args.train_size_ratio, args.sample_size
            ),
            images_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "speech_cls" or args.task_name == "all":
        task = (
            "speech_cls",
            speech_cls.get_model(),
            *speech_cls.get_train_test_dataset(
                args.seed, args.train_size_ratio, args.sample_size
            ),
            speech_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "text_cls" or args.task_name == "all":
        task = (
            "text_cls",
            text_cls.get_model(),
            *text_cls.get_train_test_dataset(
                args.seed, args.train_size_ratio, args.sample_size
            ),
            text_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if len(tasks_to_evaluate) == 0:
        raise ValueError(
            "task_name must be either 'images_cls', 'speech_cls', 'text_cls' or 'all'."
        )

    if args.grid_search:

        for task in tasks_to_evaluate:
            best_params = grid_search(task, args)

    else:
        # rerun the best parameters
        for (task_name, task_model, task_data, scoring_func) in tasks_to_evaluate:
            print("=" * 60 + f"\nRunning {args.num_runs} tests for task : {task_name}")
            # TODO: define the optimizer here

            # results[task_name] = Tester(args, task_name, task_model, task_data).run()
            if args.optimizer == "all":
                optims = list(helper.STR2OPTIM.values())
            else:
                # single optimizer
                optims = [args.optimizer]
            for optim in optims:
                tester = Tester(args, task_name, task_data, task_model, optim, params)
                tester.run()
                tester.log("./results/")


if __name__ == "__main__":
    main()
