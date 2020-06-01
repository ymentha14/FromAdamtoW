import logging
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

import helper as h


class Tester:
    def __init__(self, args, task_name, task_model, task_data):
        """Initialize the class Tester"""

        self.args = args
        self.task_name = task_name
        self.model_constructor = task_model
        self.task_data = task_data

    def train(self):
        """Perform one training on the given inputs and return the elapsed time.

        """
        if self.args.verbose:
            print("Start training ...")

        start_time = time.time()

        # 1. Construct again the model
        # self.model = self.model_constructor()

        # 3. Effectively train the model
        self._run_all_epochs()

        # 4. Store the time
        end_time = time.time()
        train_time = end_time - start_time

        if self.args.verbose:
            print("Finish training... after {:.2f}s".format(train_time))

        return train_time

    def _run_all_epochs(self):
        # TODO load from args.args ?
        NUM_EPOCHS = 100
        running_loss = 10

        for epoch in range(NUM_EPOCHS):

            running_loss -= 0.01

            if self.args.verbose:
                print(
                    "({}/{}) Training loss: {:.3f}".format(
                        epoch + 1, NUM_EPOCHS, running_loss
                    ),
                    end="\r" if epoch + 1 != NUM_EPOCHS else "\n",
                )

    def run(self):
        return self.train()
