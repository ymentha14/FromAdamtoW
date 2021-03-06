#!/usr/bin/env python3

"""
MAIN
"""

import torch

from pathlib import Path

from tasks import images_cls
from tasks import speech_cls
from tasks import text_cls
from tester import Tester

import pytorch_helper as ph
import helper as h

import random


def grid_search(task, args):
    """
    Performs the grid search. Finds the best parameters for the given task and the given grid
    Args:
        task: a tuple with
            task_name: str the name of the task (images_cls, text_cls, speech_cls)
            task_model: torch.Module
            train_dataset: torch.utils.data.DataLoader
            test_dataset: not used here!
            scoring_func: a lambda function that returns the score for a model and labeled data.
    Returns:
        best_params_per_optimizer: an object indexed by the optimizer name, with
            the best parameters found using the grid search.
    """
    (task_name, task_model, train_dataset, X, scoring_func) = task

    if args.verbose:
        print("=" * 60 + f"\nGrid Search for tasks : {task_name}")

    param_filename = h.get_param_filepath(task_name, best=False)

    print(param_filename)

    combinations = h.get_params_combinations(param_filename)

    # start of the grid search
    if args.verbose:
        print(
            "Testing {} combinations in total".format(
                sum([len(i) for i in combinations.values()])
            )
        )
        print(f"Performing grid search on {len(train_dataset)} examples.")

    best_params_per_optimizer = {}
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
                num_epochs=h.get_default_num_epochs(task_name),
            )

            # Run the cross validation phase
            (
                train_losses,
                train_accuracies,
                val_losses,
                val_accuracies,
                test_losses,
                test_accuracies,
            ) = tester.run(do_cv=True)

            # Update the best parameter combination if we specified the --overwrite_best_param argument
            best_param, best_cv_epoch, best_cv_accuracy = h.compute_best_parameter(
                val_accuracies=val_accuracies,
                best_param=best_param,
                best_cv_epoch=best_cv_epoch,
                best_cv_accuracy=best_cv_accuracy,
                param=param,
                optimizer=optim,
                verbose=True,
            )
        best_params_per_optimizer[optim] = {
            "num_epochs": best_cv_epoch,
            "param": best_param,
        }
    return best_params_per_optimizer


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
    args = h.parse_arguments()

    # Set the torch seed
    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tasks_to_evaluate = []

    if args.task_name == "images_cls" or args.task_name == "all":

        task = (
            "images_cls",
            images_cls.get_model(),
            *ph.split_train_test(
                images_cls.get_full_dataset(args.sample_size), args.train_size_ratio
            ),
            images_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "speech_cls" or args.task_name == "all":
        task = (
            "speech_cls",
            speech_cls.get_model(),
            *ph.split_train_test(
                speech_cls.get_full_dataset(args.sample_size), args.train_size_ratio
            ),
            speech_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "text_cls" or args.task_name == "all":
        task = (
            "text_cls",
            text_cls.get_model(),
            *text_cls.get_train_test_dataset(args.sample_size),
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
            if args.overwrite_best_param:
                if args.verbose:
                    print(
                        "Override best parameters for task {} with {}".format(
                            task[0], best_params
                        )
                    )
                # We want to overwrite the best parameters
                h.override_best_parameters(task[0], best_params)

        # TODO. Do w

    else:
        for task in tasks_to_evaluate:

            (task_name, task_model, train_dataset, test_dataset, scoring_func) = task

            best_params_task = h.get_best_parameters(task_name)

            if args.optimizer == "all":
                all_optimizers = h.str_2_optimizer.keys()
            else:
                all_optimizers = [args.optimizer]

            for optim_name in all_optimizers:

                best_param_optim = best_params_task[optim_name]
                best_num_epochs = best_param_optim["num_epochs"]

                print(
                    "=" * 60
                    + f"\nEvaluate {task_name} with optim {optim_name} on {args.num_runs} runs with best parameters"
                    f" {best_param_optim} for {best_num_epochs} epochs."
                )

                tester = Tester(
                    args=args,
                    task_name=task_name,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    task_model=task_model,
                    optimizer=optim_name,
                    param=best_param_optim["param"],
                    scoring_func=scoring_func,
                    num_epochs=best_num_epochs,
                )

                tester.run(num_runs=args.num_runs)


if __name__ == "__main__":
    main()
