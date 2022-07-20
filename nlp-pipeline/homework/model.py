import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers, dropout=0.5) -> None:
        super().__init__()

        self.ntokens = ntokens
        self.encoder = nn.Embedding(ntokens, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )

        self.decoder = nn.Linear(nhid, 2)
        self.embdrop = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        embedding = self.embdrop(self.encoder(input))
        output = self.rnn(embedding)[0]
        output = self.drop(output)
        output = output[-1]

        decoded = self.decoder(output)
        return decoded
