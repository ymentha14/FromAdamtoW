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

import helper as h
from src.pytorchtools import EarlyStopping


class Tester:
    """
    A tester needs to be run for one task: that is, for each dataset,optimizer and hyperparameter combination.
    """

    def __init__(
        self,
        args: object,
        task_data: torch.utils.data.DataLoader,
        task_model: nn.Module,
        optim: torch.optim,
        param: object,
        scoring_func: Callable[[nn.Module, torch.utils.data.DataLoader], float],
    ):
        """Initialize the class Tester
        
        Args:
            args: parsed arguments
            task_data: Dataloader to the dataset
            task_model: model constructor for the network that performs decently on the current dataset
            optim: torch optimizer used
            param: dict of parameters for the model/dataset/optimizer combination
        """
        self.args = args
        self.task_data = task_data
        self.model_constructor = task_model
        self.optim = optim
        self.param = param
        self.scoring_func = scoring_func
        self.device = h.get_device()

        self.patience = 10      # TODO: Is it OK?
        self.batch_size = 64  # TODO: What do we want to do with it?

    def train(self):
        """
        Perform one training on the given inputs and return the elapsed time.
        """
        if self.args.verbose:
            print("Start training ...")

        start_time = time.time()

        # 1. Construct again the model
        self.model = self.model_constructor()
        # Send it to the correct device
        self.model = self.model.to(device=self.device)  # Send model to device

        # 3. Effectively train the model
        self._run_all_epochs()

        # 4. Store the time
        end_time = time.time()
        train_time = end_time - start_time

        if self.args.verbose:
            print("Finish training... after {:.2f}s".format(train_time))
        self.train_time = train_time

        return train_time

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
        self.model.train()    # Declare we are going to train! No idea why, Pytorch stuff
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
            )  # Send data to device as tensors
            output_batch = self.model(x_batch)
            loss = criterion(output_batch, y_batch)
            total_loss += loss.item()       # Sum all losses
            self.model.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / len(dataloader.dataset)

    def compute_loss(self, model: nn.Module, loader: torch.utils.data.DataLoader, criterion: _WeightedLoss) -> float:
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
            model.eval()       # Declare we are evaluating, not training.
            losses = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = (       # Send data to device
                    x_batch.to(self.device),
                    y_batch.to(self.device),
                )
                pred = model(x_batch)     # Compute predictions
                losses += criterion(pred, y_batch).item()
            return losses / len(loader.dataset)  # Compute avg among loss and len of dataset

    def cross_validation_train_test_split(self, k: int, train_dataset: torch.utils.data.IterableDataset, cv: int):
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

    def cross_validation(self, k: int = 5, test_split: float = 0.1):
        """
        Performs k-fold cross validation on the data provided, with the model and optimizer specified.
        First it splits the dataset into test and training according to the split fraction.
        Afterwards it performs k-fold cross validation (on the training sample only).
        With the best combination of parameters found, it trains again the model and
        validates its accuracy on the test split.

        Args:
            k: int, the argument for k-cross validation.
            test_split: float, the fraction of data kept as test.
        Returns:
            Nothing
        """
        split = math.floor(len(self.task_data.dataset) * test_split)

        # Create a range with numbers from 0 to the len of the dataset-1
        indices = np.random.permutation(len(self.task_data.dataset))
        # The first indices are kept as validation, the last as training
        train_indices, val_indices = (
            np.array(indices[split:]),
            np.array(indices[:split]),
        )

        train_dataset = Subset(self.task_data.dataset, train_indices).dataset
        test_dataset = Subset(self.task_data.dataset, val_indices).dataset

        print(len(train_dataset), len(test_dataset))

        num_epochs = self.args.num_epochs
        criterion = nn.CrossEntropyLoss()
        for cv in range(k):
            # Perform another test-train split on the train_dataset
            train_loader_cv, test_loader_cv = self.cross_validation_train_test_split(k, train_dataset, cv)

            # Recreate the model (otherwise it would not start training from scratch)
            self.model = self.model_constructor()
            # Send it to the correct device
            self.model = self.model.to(device=self.device)  # Send model to device CUDA or CPU
            optimizer = self.optim(
                self.model.parameters(), **h.adapt_params(self.param)
            )
            early_stopping = EarlyStopping(patience=self.patience, verbose=True)
            val_losses = []    # Vector for validation losses (this is useful for early stopping)
            train_losses = []   # Vector for training losses (this is useful for plotting visualization)
            val_accuracies = []
            for epoch in range(num_epochs):
                # Train for one epoch, and record losses and accuracy
                train_losses.append(self._run_one_epoch(train_loader_cv, criterion, optimizer))
                val_losses.append(self.compute_loss(self.model, test_loader_cv, criterion))
                val_accuracies.append(self.score(self.model, test_loader_cv))
                early_stopping(val_losses[-1], self.model)     # Check early stopping, using the last validation loss
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Train losses: " + str(train_losses))
            print("Validation losses: " + str(val_losses))
            print("Validation accuracies: " + str(val_accuracies))
            print("\n")

        return

    def _run_all_epochs(self):
        """
        run the current model over the number of epochs specified in the args.
        """
        # 100 by default
        num_epochs = self.args.num_epochs
        # hard-coded criterion since we only use cross-entropy loss
        criterion = nn.CrossEntropyLoss()

        optimizer = self.optim(self.model.parameters(), **h.adapt_params(self.param))
        self.losses = []
        self.f1s = []

        for epoch in range(num_epochs):
            loss = self._run_one_epoch(self.task_data, criterion, optimizer)

            if self.args.verbose:
                print(
                    "({}/{}) Training loss: {:.3f}".format(epoch + 1, num_epochs, loss),
                    end="\r" if epoch + 1 != num_epochs else "\n",
                )
        print("=" * 60 + f"\nSuccess: final loss {loss}")

    def run(self):
        return self.train()

    def score(self, model: nn.Module, data: torch.utils.data.DataLoader) -> float:
        return self.scoring_func(model, data)
