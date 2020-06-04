#!/usr/bin/env python3

"""
MAIN
"""

import helper
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    results = {}

    if args.cross_validation:
        # Grid Search Mode
        for param_file, (task_name, task_model, task_data, scoring_func) in zip(
            # Get the correct param file for every task
            [helper.TASK2PARAM[t[0]] for t in tasks_to_evaluate], tasks_to_evaluate
        ):
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
                        task_data,
                        task_model,
                        helper.STR2OPTIM[optim],
                        param,
                        scoring_func,
                    )
                    # Run the cross validation phase
                    val_losses, val_accuracies, train_losses, train_accuracies = tester.cross_validation()

                    # This builds a 2 columns dataframe, one column with epoch, the other with accuracy
                    df = pd.DataFrame(val_accuracies).melt(var_name='Epochs', value_name='Accuracy')
                    accuracy_df = df.groupby('Epochs').agg({"Accuracy": ["count", "mean"]})
                    # Discard epochs that have not been reached by all cross validation attempts.
                    max_epochs_df = accuracy_df[   # count__max means that all attempts have reached such epoch
                        accuracy_df[('Accuracy', 'count')] == accuracy_df[('Accuracy', 'count')].max()]
                    best_accuracy_mean = max_epochs_df[('Accuracy', 'mean')].max()  # Get the best mean accuracy
                    best_epoch = max_epochs_df[     # Get epoch which obtained a best accuracy mean
                        max_epochs_df[('Accuracy', 'mean')] == best_accuracy_mean
                        ].index.tolist()[-1]  # Select the largest epoch with best mean (there should be only one).
                    print("Best accuracy mean: {}, obtained at epoch {}".format(best_accuracy_mean, best_epoch))
                    if best_param is None or best_cv_accuracy < best_accuracy_mean:
                        best_param = param
                        best_cv_epoch = best_epoch
                        best_cv_accuracy = best_accuracy_mean
                        print("update best param for {}:\nepochs = {}\naccuracy = {}\n params = {}".format(
                            optim,
                            best_cv_epoch,
                            best_cv_accuracy,
                            best_param
                        ))
                    else:
                        print("No improvements, best accuracy so far is {}".format(best_cv_accuracy))
                    # Do some visualization stuff here!
                    # sns.pointplot(x="Epochs", y="Accuracy",  kind='box', data=df)\
                    #     .set_title("Validation accuracy during cross validation")
                    # plt.show()
                    # and log its result
                    # tester.log(f"./results/{args.task_name}_gridsearch.json")
                print("Now we train the final model for {} using\nparams: {}\nepochs: {}"
                      .format(optim, best_param, best_cv_epoch))

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
                optims = [helper.STR2OPTIM[args.optimizer]]
            for optim in optims:
                tester = Tester(args, task_data, task_model, optim, params)
                tester.run()
                tester.log("./results/")


if __name__ == "__main__":
    main()
