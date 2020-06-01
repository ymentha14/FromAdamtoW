#!/usr/bin/env python3

"""
MAIN
"""

import argparse
import helper

import logging
from pathlib import Path
import pickle

from tasks import images_cls
from tasks import speech_cls
from tasks import text_cls

from tester import Tester


def main():
    # setting-up logs
    LOG_DIRECTORY = Path("log")
    LOG_DIRECTORY.mkdir(exist_ok=True)
    LOG_FILENAME = LOG_DIRECTORY / "results.log"

    logging.basicConfig(
        filename=LOG_FILENAME,
        filemode="a",
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b %H:%M",
        level=logging.INFO,
    )

    # Get arguments
    args = helper.parse_arguments()

    tasks_to_evaluate = []

    if args.task_name == "images_cls" or args.task_name == "all":
        task = ("images_cls", images_cls.get_model(), images_cls.get_data())
        tasks_to_evaluate.append(task)
    if args.task_name == "speech_cls" or args.task_name == "all":
        task = ("speech_cls", speech_cls.get_model(), speech_cls.get_data())
        tasks_to_evaluate.append(task)

    if args.task_name == "text_cls" or args.task_name == "all":
        task = ("text_cls", text_cls.get_model(), text_cls.get_data())
        tasks_to_evaluate.append(task)

    if len(tasks_to_evaluate) == 0:
        raise ValueError(
            "task_name must be either 'images_cls', 'speech_cls', 'text_cls' or 'all'."
        )

    results = {}

    # Run
    for (task_name, task_model, task_data) in tasks_to_evaluate:

        print("=" * 60 + f"\nRunning {args.num_runs} tests for task : {task_name}")

        results[task_name] = Tester(args, task_name, task_model, task_data).run()

    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
