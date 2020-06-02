#!/usr/bin/env python3

"""
MAIN
"""

import helper
from pathlib import Path

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

    if args.task_name == "images_cls" or args.task_name == "all":
        task = ("images_cls", images_cls.get_model(), images_cls.get_data(), images_cls.get_scoring_function())
        tasks_to_evaluate.append(task)

    if args.task_name == "speech_cls" or args.task_name == "all":
        task = ("speech_cls", speech_cls.get_model(), speech_cls.get_data(), speech_cls.get_scoring_function())
        tasks_to_evaluate.append(task)

    if args.task_name == "text_cls" or args.task_name == "all":
        task = ("text_cls", text_cls.get_model(), text_cls.get_data(), text_cls.get_scoring_function())
        tasks_to_evaluate.append(task)

    if len(tasks_to_evaluate) == 0:
        raise ValueError(
            "task_name must be either 'images_cls', 'speech_cls', 'text_cls' or 'all'."
        )
        
    # either the exploration phase or evaluation phase
    #assert((args.optimizer is None) != (args.param_file is None))

    results = {}

    if args.cross_validation:
        # Grid Search Mode
        if len(args.params_file) != len(tasks_to_evaluate):
            raise ValueError("Number of files non coherent with number of tasks to evaluate.")
        for param_file, (task_name, task_model, task_data, scoring_func) in zip(args.params_file, tasks_to_evaluate):
            print("=" * 60 + f"\nGrid Search for tasks : {task_name}")
            # create the combinations
            combinations = helper.get_params_combinations(param_file)
            # start of the grid search
            if args.verbose:
                print("Testing {} combinations in total".format(sum([len(i) for i in combinations.values()])))
            for optim, params in combinations.items():
                for param in params:
                    if args.verbose:
                        print(f"\nTesting {optim} with {param}")
                    # implement the tester
                    tester = Tester(args, task_data, task_model, helper.STR2OPTIM[optim], param, scoring_func)
                    # Run the cross validation phase
                    tester.cross_validation()
                    # and log its result
                    tester.log(f"./results/{args.task_name}_gridsearch.json")

    else:
        # rerun the best parameters
        for (task_name, task_model, task_data, scoring_func) in tasks_to_evaluate:
            print("=" * 60 + f"\nRunning {args.num_runs} tests for task : {task_name}")
            # TODO: define the optimizer here
            
            # results[task_name] = Tester(args, task_name, task_model, task_data).run()
            if args.optimizer == 'all':
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
