"""
Images classification
"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self, embedding_dim=300, hidden_dim=100, vocab_size=3000, output_size=10
    ):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embedding = self.word_embeddings(sentence)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        return self.out(lstm_out.view(len(sentence), -1))


def get_model():
    return SimpleLSTM()


"""
Load data
"""


def get_data():
    return []
