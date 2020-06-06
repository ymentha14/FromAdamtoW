import numpy as np
import torch


# Adapted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, new_score, model):

        if self.best_score is None:
            if self.verbose:
                print(f"\t\tEarlyStopping: set initial ({new_score} score)")
            self.best_score = new_score
            # self.save_checkpoint(val_loss, model)
        elif new_score < self.best_score + self.delta:
            # No improvement recorded
            self.counter += 1
            if self.verbose:
                print(
                    f"\t\tEarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(
                    f"\t\tEarlyStopping: score improve ({self.best_score} -> {new_score} score)"
                )
            self.best_score = new_score
            self.counter = 0
