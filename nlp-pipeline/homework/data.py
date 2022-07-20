from itertools import chain
import os.path
from typing import Union, overload

import torch

PAD = "<pad>"


class Dictionary:
    def __init__(self) -> None:
        self.word2idx = dict[str, int]()
        self.idx2word = list[str]()

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)

    @overload
    def __getitem__(self, idx: str) -> int:
        ...

    @overload
    def __getitem__(self, idx: int) -> str:
        ...

    def __getitem__(self, idx: Union[str, int]):
        if isinstance(idx, str):
            return self.word2idx[idx]
        if isinstance(idx, int):
            return self.idx2word[idx]
        raise TypeError("A dictionary index must be of type `str` or `int`")


class Corpus:
    def __init__(
        self, train_path: str, valid_path: str, test_path: str, seq_len: int = 30
    ) -> None:
        self.dictionary = Dictionary()
        self.seq_len = seq_len

        self.dictionary.add_word(PAD)

        self.train = self.tokenize(train_path)
        self.valid = self.tokenize(valid_path)
        self.test = self.tokenize(test_path)

    def tokenize(self, path: str):
        assert os.path.exists(path)

        with open(path, "r", encoding="utf8") as f:
            return [self.process_line(line) for line in f]

    def process_line(self, line: str):
        line, label = line.split("\t")
        splitted = line.split()

        for word in splitted:
            self.dictionary.add_word(word)

        line_id = (
            torch.tensor(
                self.padded([self.dictionary[word] for word in splitted])
            ).type(torch.int32),
            self.label(label.strip()),
        )

        return line_id

    def padded(self, seq: list[int]) -> list[int]:
        l = len(seq)
        if l < self.seq_len:
            return seq + [self.dictionary[PAD]] * (self.seq_len - l)
        if l > self.seq_len:
            return seq[: self.seq_len]
        return seq

    @staticmethod
    def label(label_text: str):
        if label_text == "positive":
            return 1
        if label_text == "negative":
            return 0
        raise ValueError(f"Wrong label '{label_text}'")
