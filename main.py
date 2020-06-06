#!/usr/bin/env python3

"""
MAIN
"""
import math

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

import helper
from pathlib import Path
import numpy as np

from tasks import images_cls
from tasks import speech_cls
from tasks import text_cls
from tester import Tester


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

    tasks_to_evaluate = []

    size_of_dataset = args.size_dataset_sample

    if args.task_name == "images_cls" or args.task_name == "all":
        task = (
            "images_cls",
            images_cls.get_model(),
            images_cls.get_data(size_of_dataset),
            images_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "speech_cls" or args.task_name == "all":
        task = (
            "speech_cls",
            speech_cls.get_model(),
            speech_cls.get_data(size_of_dataset),
            speech_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if args.task_name == "text_cls" or args.task_name == "all":
        task = (
            "text_cls",
            text_cls.get_model(),
            text_cls.get_data(size_of_dataset),
            text_cls.get_scoring_function(),
        )
        tasks_to_evaluate.append(task)

    if len(tasks_to_evaluate) == 0:
        raise ValueError(
            "task_name must be either 'images_cls', 'speech_cls', 'text_cls' or 'all'."
        )

    # either the exploration phase or evaluation phase
    # assert((args.optimizer is None) != (args.param_file is None))

    if args.cross_validation:
        # CROSS VALIDATION
        for param_file, (task_name, task_model, task_data, scoring_func) in zip(
            # Get the correct param file for every task
            [helper.TASK2PARAM[t[0]] for t in tasks_to_evaluate],
            tasks_to_evaluate,
        ):
            test_split = args.train_test_split
            # Split test and train data
            split = math.floor(len(task_data.dataset) * test_split)

            # Create a range with numbers from 0 to the len of the dataset-1
            indices = np.random.permutation(len(task_data.dataset))
            # The first indices are kept as validation, the last as training
            train_indices, val_indices = (
                np.array(indices[split:]),
                np.array(indices[:split]),
            )

            train_dataloader = DataLoader(
                Subset(task_data.dataset, train_indices), batch_size=args.batch_size
            )  # TODO: fix batch size
            test_dataloader = DataLoader(
                Subset(task_data.dataset, val_indices), batch_size=args.batch_size
            )

            print("=" * 60 + f"\nGrid Search for tasks : {task_name}")
            # create the combinations
            combinations = helper.get_params_combinations(param_file)
            # start of the grid search
            if args.verbose:
                print(
                    "Testing {} combinations in total".format(
                        sum([len(i) for i in combinations.values()])
                    )
                )
                print(
                    "Len of training dataset: {}\nLen of validation dataset: {}".format(
                        len(test_dataloader.dataset), len(train_dataloader.dataset)
                    )
                )
            for optim, params in combinations.items():
                best_param = None
                best_cv_accuracy = None
                best_cv_epoch = None
                for param in params:
                    if args.verbose:
                        print(f"\nTesting {optim} with {param}")
                    # implement the tester
                    tester = Tester(
                        args,
                        task_name,
                        train_dataloader,
                        task_model,
                        optim,
                        param,
                        scoring_func,
                    )
                    # Run the cross validation phase
                    (
                        val_losses,
                        val_accuracies,
                        train_losses,
                        train_accuracies,
                    ) = tester.cross_validation()

                    # Update the best parameter combination, if the accuracy for this cross validation phase is higher
                    (
                        best_param,
                        best_cv_epoch,
                        best_cv_accuracy,
                    ) = helper.get_best_parameter(
                        val_accuracies,
                        best_param,
                        best_cv_accuracy,
                        best_cv_epoch,
                        param,
                        optim,
                        True,
                    )

                    # and log its result
                    # tester.log(f"./results/{args.task_name}_gridsearch.json")
                print(
                    "Now we train the final model for {} using\nparams: {}\nepochs: {}".format(
                        optim, best_param, best_cv_epoch
                    )
                )
                # Train the model using the best hyper parameters found so far using cross validation
                tester = Tester(
                    args,
                    task_name,
                    train_dataloader,
                    task_model,
                    optim,
                    best_param,
                    scoring_func,
                )
                result = tester.train(test_dataloader, best_cv_epoch)
                helper.log_results(
                    result,
                    best_cv_epoch,
                    best_param,
                    optim,
                    helper.TASK2LOGFILE[task_name],
                )
                # Test on the test data.
                print(
                    "The score on the validation data for the best model found is: {}".format(
                        scoring_func(tester.model, test_dataloader)
                    )
                )

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
