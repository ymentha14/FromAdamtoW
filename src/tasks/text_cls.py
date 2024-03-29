"""
Text classification
"""

import torch
import torch.nn as nn

from torch.utils.data.dataset import Subset
from torchtext.datasets import text_classification

import pytorch_helper as ph
import helper


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=95812,  # TODO len(train_dataset.get_vocab())
        embed_dim=64,
        num_classes=4,
        hidden_dim=8,
        n_layers=2,
        bidirectional=True,
        dropout=0.1,
        rnn_type="lstm",
        with_rnn=False,
    ):

        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )

        self.fc_with_rnn = nn.Linear(hidden_dim * 2, num_classes)
        self.fc_with_embed = nn.Linear(embed_dim, num_classes)

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

        self.with_rnn = with_rnn

    def forward(self, x_batch):

        # (text, text_lengths) = x_batch

        text_lengths = x_batch[:, 0]
        text = x_batch[:, 1:]

        # text = [batch size, sent_length]
        embedded = self.embedding(text)

        if self.with_rnn:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths, batch_first=True, enforce_sorted=False
            )

            packed_output, (hidden, cell) = self.rnn(packed_embedded)

            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

            out = self.fc_with_rnn(hidden)

        else:
            embedded = embedded.mean(dim=1)
            out = self.fc_with_embed(embedded)

        return out


def get_model():
    return TextClassifier


def get_scoring_function():
    """
    Returns the function that computes the score, given the model and the data.
    Returns:
        score_func: (model: nn.Module, data: torch.utils.data.DataLoader) -> float
    """

    def accuracy(model: nn.Module, data: torch.utils.data.DataLoader):
        device = ph.get_device()
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


"""
Load data
"""


def get_train_test_dataset(sample_size: int = None):
    """
    Return Dataset

    Args:
        sample_size: int, take a sample of smaller size in order to train faster. None: take all sample
    Returns:
        dataset: of type DataLoader
    """

    #
    NGRAMS = 1

    # create data folder in case it does not exist
    import os

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    train_dataset, test_dataset = text_classification.DATASETS["AG_NEWS"](
        root="./data", ngrams=NGRAMS, vocab=None
    )

    if sample_size is not None:
        train_dataset = helper.get_sample(sample_size, train_dataset)

    return train_dataset, test_dataset
