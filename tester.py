import math
import time
import json
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data.sampler import SubsetRandomSampler

import helper
import helper as h
from src.pytorchtools import EarlyStopping


class Tester:
    """
    A tester needs to be run for one task: that is, for each dataset,optimizer and hyperparameter combination.
    """

    def __init__(
        self,
        args: object,
        task_name: str,
        task_data: torch.utils.data.DataLoader,
        task_model: nn.Module,
        optimizer: str,
        param: object,
        scoring_func: Callable[[nn.Module, torch.utils.data.DataLoader], float],
    ):
        """Initialize the class Tester
        
        Args:
            args: parsed arguments
            task_name: name of the task
            task_data: Dataloader to the dataset
            task_model: model constructor for the network that performs decently on the current dataset
            optim: torch optimizer used
            param: dict of parameters for the model/dataset/optimizer combination
        """
        self.args = args
        self.task_name = task_name
        self.task_data = task_data
        self.model_constructor = task_model
        self.optim = h.STR2OPTIM[optimizer]
        self.optim_name = optimizer
        self.param = param
        self.scoring_func = scoring_func
        self.device = h.get_device()

        self.patience = 10  # TODO: Is it OK?
        self.batch_size = self.args.batch_size  # TODO: What do we want to do with it?

    def train(self, test_data: torch.utils.data.DataLoader, epochs: int = None):
        """
        Perform one training on the given inputs and returns an object with the following format:
        {
            "train_accuracy": [0.9, ...],
            "train_loss": [0.5,..],
            "time_elapsed": 10.0
        }
        Args:
            test_data: data loader with testing data.
            epochs: int, the number of epochs the models has to be trained for. If None, then use as parameter
                the command line argument num_epochs, if it is specified then do otherwise.
                (it is specified, for instance, in case we have performed cross validation and we already know
                what is the best parameter).
        Returns:
            results: an object with the above format
        """

        epochs = epochs if epochs is not None else self.args.num_epochs
        if self.args.verbose:
            print("Start training ...")

        start_time = time.time()

        # 1. Construct again the model
        self.model = self.model_constructor()
        # Send it to the correct device
        self.model = self.model.to(device=self.device)  # Send model to device

        # 3. Effectively train the model
        (
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
        ) = self._run_all_epochs(epochs, test_data)

        # 4. Store the time
        end_time = time.time()
        train_time = end_time - start_time

        if self.args.verbose:
            print("Finish training... after {:.2f}s".format(train_time))
        self.train_time = train_time

        return {
            "train_accuracy": train_accuracies,
            "train_loss": train_losses,
            "val_accuracy": val_accuracies,
            "val_loss": val_losses,
            "time_elapsed": train_time,
        }

    def log(self, log_path: str):
        """append the scores of the current run to the json in log_path"""

        log_path_posix = Path(log_path)
        if not log_path_posix.exists():
            with open(log_path, "w") as f:
                json.dump({}, f, indent=4)
        date = datetime.now().strftime("%m_%d_%y-%H_%M")
        log_data = copy(self.param)
        log_data["optim"] = str(self.optim)
        log_data["losses"] = self.losses
        log_data["train_time"] = self.train_time
        new_data = {date: log_data}
        # old_log = json.loads(log_path)

        with open(log_path, "r") as f:
            old_log = json.load(f)

        old_log.update(new_data)
        with open(log_path, "w") as f:
            json.dump(old_log, f, indent=4)

    def _run_one_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: _WeightedLoss,
        optimizer: torch.optim,
    ) -> float:
        """
        Run through all batches in the input dataset, and perform forward and backward pass.
        Args:
            dataloader: the data loader upon which to compute one epoch (training data)
            criterion: the loss function
            optimizer: the optimizer used (Adam, SGD, AdamW)
        Returns:
            loss: float, the loss of the training dataset (averaged on the length of the dataset itself)
        """
        self.model.train()  # Declare we are going to train! No idea why, Pytorch stuff
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
            )  # Send data to device as tensors
            output_batch = self.model(x_batch)
            loss = criterion(output_batch, y_batch)
            total_loss += loss.item()  # Sum all losses
            self.model.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / len(dataloader.dataset)

    def compute_loss(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: _WeightedLoss,
    ) -> float:
        """
        Compute the loss by summing the loss of all batches
        Args:
            model: the model used to compute the loss
            loader: the testing data
            criterion: the loss function
        Returns:
            loss: the average loss given by criterion(loader)/len(dataset)
        """
        with torch.no_grad():  # Deactivate gradient recording!
            model.eval()  # Declare we are evaluating, not training.
            losses = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = (  # Send data to device
                    x_batch.to(self.device),
                    y_batch.to(self.device),
                )
                pred = model(x_batch)  # Compute predictions
                losses += criterion(pred, y_batch).item()
            return losses / len(
                loader.dataset
            )  # Compute avg among loss and len of dataset

    def cross_validation_train_test_split(
        self, k: int, train_dataset: torch.utils.data.Dataset, cv: int
    ):
        """
        Splits the dataset into 2 partitions:
        one goes from len(train_dataset) * (split * cv) to len(train_dataset) * (split * (cv+1)) (the validation one)
        and the other takes all the rest (the training one).
        Args:
            k: k-fold cross validation parameter
            train_dataset: the dataset to split
            cv: the cross validation iterator (from 0 to k-1)
        Returns:
            A training and a validation data loader
        """
        split = 1 / k
        indices = np.random.permutation(len(train_dataset))
        lower_extreme = math.floor(len(train_dataset) * (split * cv))
        higher_extreme = math.floor(len(train_dataset) * (split * (cv + 1)))
        train_indices_cv = np.array(indices[lower_extreme:higher_extreme])
        # Merge the complement of train_indices_cv with respect of train_indices
        test_indices_cv = np.concatenate(
            [indices[:lower_extreme], indices[higher_extreme:]]
        )

        train_dataset_cv = Subset(train_dataset, train_indices_cv)
        test_dataset_cv = Subset(train_dataset, test_indices_cv)
        train_loader_cv = torch.utils.data.DataLoader(
            train_dataset_cv, batch_size=self.batch_size
        )
        test_loader_cv = torch.utils.data.DataLoader(
            test_dataset_cv, batch_size=self.batch_size
        )
        return train_loader_cv, test_loader_cv

    def cross_validation(self, k: int = 5):
        """
        Performs k-fold cross validation on the data provided, with the model and optimizer specified.
        First it splits the dataset into test and training according to the split fraction.
        Afterwards it performs k-fold cross validation (on the training sample only).
        With the best combination of parameters found, it trains again the model and
        validates its accuracy on the test split.

        Args:
            k: int, the argument for k-cross validation.
        Returns:
            val_losses: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with validation losses obtained in every k-th-attempt,  in every epoch
            val_accuracies: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with validation accuracies obtained in every k-th-attempt,  in every epoch
            train_losses: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with training losses obtained in every k-th-attempt,  in every epoch
            train_accuracies: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with training accuracies obtained in every k-th-attempt,  in every epoch
        """

        num_epochs = self.args.num_epochs
        criterion = nn.CrossEntropyLoss()

        val_losses = []
        val_accuracies = []
        train_losses = []
        train_accuracies = []
        for cv in range(k):
            # Perform another test-train split on the train_dataset
            train_loader_cv, test_loader_cv = self.cross_validation_train_test_split(
                k, self.task_data.dataset, cv
            )

            # Recreate the model (otherwise it would not start training from scratch)
            self.model = self.model_constructor()
            # Send it to the correct device
            self.model = self.model.to(
                device=self.device
            )  # Send model to device CUDA or CPU
            optimizer = self.optim(
                self.model.parameters(), **h.adapt_params(self.param)
            )
            early_stopping = EarlyStopping(patience=self.patience, verbose=False)
            val_losses_cv = (
                []
            )  # Vector for validation losses (this is useful for early stopping)
            train_losses_cv = (
                []
            )  # Vector for training losses (this is useful for plotting visualization)
            val_accuracies_cv = []
            train_accuracies_cv = []
            for epoch in range(num_epochs):
                # Train for one epoch, and record losses and accuracy
                train_losses_cv.append(
                    self._run_one_epoch(train_loader_cv, criterion, optimizer)
                )
                val_losses_cv.append(
                    self.compute_loss(self.model, test_loader_cv, criterion)
                )
                val_accuracies_cv.append(self.score(self.model, test_loader_cv))
                train_accuracies_cv.append(self.score(self.model, train_loader_cv))
                # TODO: what do we want to use? Accuracy or Loss? Now it is accuracy, but maybe it is better loss
                early_stopping(
                    val_accuracies_cv[-1], self.model
                )  # Check early stopping, using last val accuracy
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # print("Training losses: " + str(train_losses_cv))
            # print("Validation losses: " + str(val_losses_cv))
            # print("Validation accuracies: " + str(val_accuracies_cv))
            # print("Training accuracies: " + str(train_accuracies_cv))
            val_losses.append(val_losses_cv)
            val_accuracies.append(val_accuracies_cv)
            train_accuracies.append(train_accuracies_cv)
            train_losses.append(train_losses_cv)

        helper.log_results_cross_validation(
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
            self.optim_name,
            helper.TASK2LOGFILE[self.task_name],
        )
        return (
            np.array([np.array(el) for el in val_losses]),
            np.array([np.array(el) for el in val_accuracies]),
            np.array([np.array(el) for el in train_losses]),
            np.array([np.array(el) for el in train_accuracies]),
        )

    def _run_all_epochs(self, num_epochs: int, test_data: torch.utils.data.DataLoader):
        """
        run the current model over the number of epochs specified as parameter.
        Args:
            num_epochs: int, the number of epochs used to train the model.
            test_data: the test data loader
        Returns:
            train_losses: a list of training loss for each epoch
            train_accuracies: a list of accuracies for each epoch
            val_losses: a list of validation loss for each epoch
            val_accuracies: a list of validation accuracies for each epoch
        """
        # hard-coded criterion since we only use cross-entropy loss
        criterion = nn.CrossEntropyLoss()

        optimizer = self.optim(self.model.parameters(), **h.adapt_params(self.param))
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            train_losses.append(
                self._run_one_epoch(self.task_data, criterion, optimizer)
            )
            train_accuracies.append(self.scoring_func(self.model, self.task_data))
            val_losses.append(self.compute_loss(self.model, test_data, criterion))
            val_accuracies.append(self.scoring_func(self.model, test_data))

        return train_losses, train_accuracies, val_losses, val_accuracies

    def run(self):
        return self.train()

    def score(self, model: nn.Module, data: torch.utils.data.DataLoader) -> float:
        return self.scoring_func(model, data)
