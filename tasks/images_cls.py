"""
Images classification
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

import helper


class Cnn(nn.Module):
    # The network is a CNN, with one convolutional layer, dropout, and 2 fully connected layers
    def __init__(self, dropout: float = 0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(
            1, 32, 5, 1
        )  # Kernel size of 5, TODO: it may be a good parameter to optimize with
        self.dropout1 = nn.Dropout2d(dropout)
        # The size is computed as 32 (filters) * 24 (height after applying kernel) * 24 (width after applying kernel)
        self.fc1 = nn.Linear(18432, 500)
        self.dropout2 = nn.Dropout2d(dropout)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_model():
    """
    Returns the model, in this case a CNN
    Returns:
        model: of type nn.Module
    """
    return Cnn


def get_data():
    """
    Return a DataLoader for the training data.
    Returns:
        dataset: of type DataLoader
    """
    # Initialize the training loader, use special parameters if cuda is available.
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # Parameters advised by Torch documentation
                ]
            ),
        ),
        batch_size=64,
        shuffle=True,
        **kwargs
    )
    return train_loader


def get_scoring_function():
    """
    Returns the function that computes the score, given the model and the data (as a torch DataLoader).
    In case of images_cls the scoring function is the accuracy (correct / total).
    Returns:
        score_func: (model: nn.Module, data: torch.utils.data.DataLoader) -> float
    """

    def accuracy(model: nn.Module, data: torch.utils.data.DataLoader):
        device = helper.get_device()
        model.eval()
        model.to(device=device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        return 100.0 * correct / total

    return accuracy
