import math
import time
import json
from copy import copy
from typing import Callable, Optional, List


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data.sampler import SubsetRandomSampler

import helper as h
import pytorch_helper as ph
from pytorch_helper import EarlyStopping


class Tester:
    """
    A tester needs to be run for one task: that is, for each dataset,optimizer and hyperparameter combination.
    """

    def __init__(
        self,
        args: object,
        task_name: str,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        task_model: nn.Module,
        optimizer: str,
        param: object,
        scoring_func: Callable[[nn.Module, torch.utils.data.DataLoader], float],
        num_epochs: int,
    ):
        """Initialize the class Tester
        
        Args:
            args: parsed arguments
            task_name: name of the task
            train_dataset: train dataset
            test_dataset: test dataset
            task_model: model constructor for the network that performs decently on the current dataset
            optimizer: name of the optimizer used (Adam, AdamW, SGD)
            param: dict of parameters for the model/dataset/optimizer combination
            scoring_func: Callable, a function that given a model and a dataloader (hence with labels)
                returns the score (accuracy).
            num_epochs: max number of epochs the model is going to be trained for. (if do_cv is true when performing run)
                then num_epochs is the max set by the command line argument --num_epochs. If do_cv is false, then use
                the num_epochs specified in the file best_param).
        """
        self.args = args
        self.task_name = task_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_constructor = task_model

        self.optim_name = optimizer
        self.param = param
        self.compute_score = lambda dataloader: scoring_func(self.model, dataloader)

        self.device = ph.get_device()
        self.patience = args.patience
        self.batch_size = args.batch_size  # TODO make it custom for each task.

        self.criterion_constructor = nn.CrossEntropyLoss
        self.optimizer_constructor = h.str_2_optimizer[
            optimizer
        ]  # TODO we can do it early

        self.num_epochs = num_epochs

    def compute_loss(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Compute the loss by summing the loss of all batches
        Args:
            dataloader: the data upon which to compute the loss
        Returns:
            loss: the average loss given by criterion(dataloader)/len(dataloader).
                Note that actually the len of the dataloader is the number of batches, not the len of the dataset.
        """

        assert type(dataloader) == torch.utils.data.DataLoader

        with torch.no_grad():  # Deactivate gradient recording!
            self.model.eval()  # Declare we are evaluating, not training.
            losses = 0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = (  # Send data to device
                    x_batch.to(self.device),
                    y_batch.to(self.device),
                )
                pred = self.model(x_batch)  # Compute predictions
                losses += self.criterion(pred, y_batch).item()
            return losses / len(
                dataloader.dataset
            )  # Compute avg among loss and len of dataset

    def cross_validation(self, kfold: bool):
        """
        Performs k-fold cross validation on the data provided, on the model and optimizer specified.
        Can be done either by performing k-fold cross validation, either by splitting k times the test and train data
        at random.

        Args:
            kfold: bool: set it to true if you want to perform k-fold cross validation.
            ... 
        Returns:
            val_losses: 2-dimensional array 
            (k x num_epochs, second dimension may vary due to early stopping),
                with validation losses obtained in every k-th-attempt,  in every epoch
            val_accuracies: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with validation accuracies obtained in every k-th-attempt,  in every epoch
            train_losses: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with training losses obtained in every k-th-attempt,  in every epoch
            train_accuracies: 2-dimensional array (k x num_epochs, second dimension may vary due to early stopping),
                with training accuracies obtained in every k-th-attempt,  in every epoch
        """

        num_epochs = self.num_epochs

        train_losses = []
        train_accuracies = []

        val_losses = []
        val_accuracies = []

        if kfold:
            split = ph.split_kfold(
                dataset=self.train_dataset,
                k=self.args.k,
                batch_size=self.batch_size,
                task_name=self.task_name,
            )
        else:
            split = ph.split_k_times(
                dataset=self.train_dataset,
                k=self.args.k,
                batch_size=self.batch_size,
                train_ratio=self.args.train_size_ratio,
                task_name=self.task_name,
            )

        for i, (train_dataloader_cv, val_dataloader_cv) in enumerate(split):

            if self.args.verbose:
                print(f"\tRun {i}-split.")

            (
                train_losses_cv,
                train_accuracies_cv,
                val_losses_cv,
                val_accuracies_cv,
            ) = self._train(
                train_dataloader=train_dataloader_cv,
                val_dataloader=val_dataloader_cv,
                with_early_stopping=True,
            )

            train_losses.append(train_losses_cv)
            train_accuracies.append(train_accuracies_cv)

            val_losses.append(val_losses_cv)
            val_accuracies.append(val_accuracies_cv)

        return (
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
        )

    def _run_one_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Run through all batches in the input dataset, and perform forward and backward pass.
        Args:
            dataloader: the data loader upon which to compute one epoch (training data)
        Returns:
            loss: float, the loss of the training dataset (averaged on the length of the dataset itself)
        """
        # Set model into training mode
        self.model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = (
                x_batch.to(self.device),
                y_batch.to(self.device),
            )  # Send data to device as tensors
            output_batch = self.model(x_batch)
            loss = self.criterion(output_batch, y_batch)
            total_loss += loss.item()  # Sum all losses
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(dataloader.dataset)

    def _train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        with_early_stopping: bool,
    ):
        """
        Effectively train the model

        Say that val_dataloader is optional.

        Args:
            train_dataloader: training dataloader
            val_dataloader: testing dataloader
            with_early_stopping: Stop the model using early stopping criterion. We don't want to do so when we are
                training the final model with best parameters (the number of epochs is fixed in this case).
        Returns:
            train_losses: a list of training loss for each epoch
            train_accuracies: a list of accuracies for each epoch
            val_losses: a list of validation loss for each epoch
            val_accuracies: a list of validation accuracies for each epoch
        """

        assert type(train_dataloader) == torch.utils.data.DataLoader

        # Recreate the model
        model = self.model_constructor()

        # Send model to CUDA or CPU
        self.model = model.to(self.device)

        self.criterion = self.criterion_constructor()

        self.optimizer = self.optimizer_constructor(
            model.parameters(), **h.adapt_params(self.param)
        )

        train_losses = []
        train_accuracies = []

        if val_dataloader:
            val_losses = []
            val_accuracies = []

        early_stopping = EarlyStopping(
            patience=self.patience, verbose=self.args.verbose
        )

        for epoch in range(self.num_epochs):

            loss = self._run_one_epoch(train_dataloader)
            train_losses.append(loss)

            train_accuracies.append(self.compute_score(train_dataloader))

            if val_dataloader:
                val_losses.append(self.compute_loss(val_dataloader))

                val_accuracy_cv = self.compute_score(val_dataloader)
                val_accuracies.append(val_accuracy_cv)

            if with_early_stopping:
                early_stopping(
                    val_accuracy_cv, self.model,f"({epoch+1}/{self.num_epochs}):"
                )  # Check early stopping, using last val accuracy
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if val_dataloader:
            return train_losses, train_accuracies, val_losses, val_accuracies
        return train_losses, train_accuracies

    def run(self, do_cv: bool = False) -> Optional[List[int]]:
        """
        Run the tester.

        By default, train on the whole train_dataset and test on the test_dataset and store the changes in the log.

        If do_cv is set to True, train with cross_validation and return the mean of the loss and the score, as well as the optimal number of epochs.
        Args:
            do_cv: boolean, set to True if you want to perform cross validation, otherwise it will run the
                model once (usually, using the best parameters found with grid search)
        """

        if not do_cv:

            train_dataloader = ph.get_dataloader(
                self.train_dataset, self.batch_size, self.task_name
            )
            test_dataloader = ph.get_dataloader(
                self.test_dataset, self.batch_size, self.task_name
            )
            train_losses, train_accuracies, test_losses, test_accuracies = self._train(
                train_dataloader=train_dataloader,
                val_dataloader=test_dataloader,  # only used for computing statistics, not early stopping
                with_early_stopping=False,
            )

            val_losses = None
            val_accuracies = None
        else:
            (
                train_losses,
                train_accuracies,
                val_losses,
                val_accuracies,
            ) = self.cross_validation(kfold=False)

            test_losses = None
            test_accuracies = None

        return (
            train_losses,
            train_accuracies,
            val_losses,
            val_accuracies,
            test_losses,
            test_accuracies,
        )
