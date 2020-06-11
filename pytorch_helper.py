import torch
import numpy as np


def split_train_test(dataset, train_ratio):
    """
    Split dataset into two parts
    """

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def split_kfold(dataset, k, batch_size, task_name):
    """
    Split the dataset with k-fold

    Return an iterable (generator) where each element is a tuple (train_dataloader, val_dataloader) with batch_size batch_size
    """

    kfold = KFold(n_splits=k)

    for train_index, val_index in kfold.split(dataset):

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)

        train_dataloader = get_dataloader(train_dataset, batch_size, task_name)

        val_dataloader = get_dataloader(val_dataset, batch_size, task_name)

        yield (train_dataloader, val_dataloader)


def split_k_times(dataset, k, batch_size, train_ratio, task_name):
    """
    Split the dataset k-times with a given train ratio

    Return an iterable (generator)
    """

    for _ in range(k):
        train_dataset, val_dataset = split_train_test(dataset, train_ratio)

        train_dataloader = get_dataloader(train_dataset, batch_size, task_name)

        val_dataloader = get_dataloader(val_dataset, batch_size, task_name)

        yield (train_dataloader, val_dataloader)


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

    def __call__(self, new_score, model, epoch=""):
        """
        epoch (str): information about the epoch as '(n/N)' with n the current run and N the total runs
        """

        if self.best_score is None:
            if self.verbose:
                print(f"\t\t{epoch}EarlyStopping: set initial ({new_score} score)")
            self.best_score = new_score
            # self.save_checkpoint(val_loss, model)
        elif new_score <= self.best_score + self.delta:
            # No improvement recorded
            self.counter += 1
            if self.verbose:
                print(
                    f"\t\t{epoch}EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(
                    "\t\t{}EarlyStopping: score improve ({:.2f} -> {:.2f} score)".format(
                        epoch, self.best_score, new_score
                    )
                )
            self.best_score = new_score
            self.counter = 0


def get_device():
    """
    Get the device, CUDA or CPU depending on the machine availability.
    Returns:
        device: CUDA or CPU
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def generate_batch_text_cls(batch):

    sequences = [seq for (label, seq) in batch]
    pad_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    seq_length = torch.tensor([len(seq) for (label, seq) in batch]).reshape(-1, 1)
    labels = torch.tensor([label for (label, seq) in batch])

    x = torch.cat([seq_length, pad_sequences], dim=1)

    return x, labels


def get_dataloader(
    dataset: torch.utils.data.Dataset, batch_size, task_name, shuffle=True
):
    """
    Return a dataloader given a dataset. Task name should be specified as some task (for instance text),
    need special treatments.
    """

    collate_fn = None
    if task_name == "text_cls":
        collate_fn = generate_batch_text_cls

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
