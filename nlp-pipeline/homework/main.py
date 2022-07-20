import argparse
import math
import os
import time

import torch
import torch.nn as nn

from data import Corpus
from model import RNNModel


def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch glue-sst2 RNN/LSTM/GRU/Transformer Language Model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../data/glue-sst2",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--seqlen", type=int, default=30, help="length of sequence in glue-sst2"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RNN_RELU",
        help="type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)",
    )
    parser.add_argument(
        "--emsize", type=int, default=200, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="initial learning rate"  # 你可能需要调整它
    )
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=128, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, default=4242, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )
    args = parser.parse_args()
    return args


def batchify(
    data: list[tuple[torch.Tensor, int]], batch_size: int, device: torch.device
):
    return [
        (
            torch.nn.utils.rnn.pad_sequence(batch[0]).to(device),  # type: ignore
            torch.tensor(
                batch[1],
            ).to(device),
        )
        for i in range(0, len(data), batch_size)
        if (batch := tuple(zip(*data[i : i + batch_size], strict=True)))
    ]


def evaluate(model, data, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (batch, target) in enumerate(data):
            output = model(batch)
            pred = torch.argmax(output, dim=1)
            total_loss += criterion(output[: len(target)], target).item()
            correct += (pred[: len(target)] == target).float().mean()
    return total_loss / len(data), correct / len(data)


def train(
    model,
    train_data,
    criterion,
    optimizer,
    epoch,
    clip,
    log_interval,
):
    model.train()
    total_loss = 0
    start_time = time.time()
    for i, (batch, target) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output[: len(target)], target)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip)

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            start_time = time.time()
            total_loss = 0
            print(
                "| epoch {:3d} | {:5d} / {:5d} batches | {:7.2f} ms/batch | "
                "loss {:7.4f} |".format(
                    epoch,
                    i,
                    len(train_data),
                    elapsed * 1000 / log_interval,
                    cur_loss,
                )
            )


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basepath = args.data
    train_path = os.path.join(basepath, "train.txt")
    test_path = os.path.join(basepath, "test.txt")
    valid_path = os.path.join(basepath, "dev.txt")

    corpus = Corpus(train_path, valid_path, test_path, args.seqlen)
    batch_size = args.batch_size
    train_data = batchify(corpus.train, batch_size, device)
    test_data = batchify(corpus.test, batch_size, device)
    valid_data = batchify(corpus.valid, batch_size, device)

    ntokens = len(corpus.dictionary)
    model = RNNModel(
        args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-5)

    best_val_loss = 0
    print("-" * 71)
    for epoch in range(args.epochs):
        epoch += 1
        epoch_start_time = time.time()
        train(
            model,
            train_data,
            criterion,
            optimizer,
            epoch,
            args.clip,
            args.log_interval,
        )
        val_loss, val_acc = evaluate(model, valid_data, criterion)
        print("-" * 71)
        print(
            "| end of epoch {:2d} | time: {:4.2f}s | valid loss {:5.4f} | valid acc {:5.2f} |".format(
                epoch,
                time.time() - epoch_start_time,
                val_loss,
                val_acc,
            )
        )
        print("-" * 71)
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model, args.save)
            best_val_loss = val_loss

    model = torch.load(args.save)
    test_loss, test_acc = evaluate(model, test_data, criterion)
    print(
        "\nEnd of training:\ntest loss {:5.4f}\ntest acc {:5.2f}%".format(
            test_loss, test_acc, math.exp(test_loss)
        )
    )


if __name__ == "__main__":
    main()
