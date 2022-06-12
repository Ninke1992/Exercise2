from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
from scipy.io import arff
from torch.nn.utils.rnn import pad_sequence

Tensor = torch.Tensor


class BaseDataset:
    def __init__(self, datapath: Path) -> None:
        self.path = datapath
        self.data = self.process_data()

    def process_data(self) -> List[Tuple[int, Tensor]]:
        data = arff.loadarff(self.path)  # read loaded arff file
        first_label = int(data[0][0][14])  # get the first label
        label = first_label  # set first label to var label
        chunck = []  # empty list to store chunck items
        chuncks = []  # empty list to store chuncks
        for line in data[0]:  # loop over file line by line
            if (
                int(line[14]) == label
            ):  # if the line has same label as previous line label
                observation = []
                for index, i in enumerate(line):  # loop over items in line
                    if index != 14:  # add all except for element 14, which is label
                        observation.append(i)
                observationTensor = torch.Tensor(observation)  # make tensor of line
                chunck.append(observationTensor)  # add tensor to chunck
            else:  # if line doesn't have same label as previous line label
                chunck_tuple = (
                    label,
                    torch.stack(chunck),
                )  # make a tuple of current label and matrix of chunck of tensors
                chuncks.append(chunck_tuple)  # add this tuple to list of chuncks
                label = int(line[14])  # set new current label
                chunck = []  # init empty list for new chunck
                observation = []  # empty list for new line
                for index, i in enumerate(line):  # loop over items in line
                    if index != 14:  # add all except for element 14, which is label
                        observation.append(i)
                observationTensor = torch.Tensor(observation)
                chunck.append(observationTensor)
        chunck_tuple = (
            label,
            torch.stack(chunck),
        )  # after all lines are looped through, add last chunck to list of chuncks
        chuncks.append(chunck_tuple)
        return chuncks

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        item = self.data[idx]
        x = item[1]
        y = item[0]
        return x, y

    def __len__(self) -> int:
        length = len(self.data)
        return length


class BaseDataIterator:
    def __init__(self, dataset: BaseDataset, window_size: int, batchsize: int) -> None:
        self.dataset = dataset  # set dataset
        self.batchsize = batchsize  # set batchsize
        min_length = dataset[0][0].shape[0]
        for i in range(len(dataset)):
            lenght = dataset[i][0].shape[0]
            if lenght < min_length:
                min_length = lenght
        if window_size > min_length:
            print(
                """Maximum window length is {}, setting window length to {}.
                Use PaddedDataIterator for bigger window size""".format(
                    min_length, min_length
                )
            )
            self.window_size = min_length
        else:
            self.window_size = window_size  # set windowsize

    def __iter__(self) -> BaseDataIterator:
        self.index = 0
        count = 0
        for index, _ in enumerate(self.dataset):
            observation = self.dataset.__getitem__(index)
            chuncks = torch.split(observation[0], self.window_size)
            nr_chuncks = math.ceil(len(chuncks))
            count += nr_chuncks
        self.index_list = torch.randperm(count)
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.index_list) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            return torch.stack(X), torch.Tensor(Y)
        else:
            raise StopIteration

    def get_chuck(self) -> Tuple[List, int]:
        i = random.randint(
            0, len(self.dataset) - 1
        )  # get random number between 0 and length of chuncks
        observation = self.dataset.__getitem__(i)  # get the item from dataset
        chuncks = torch.split(
            observation[0], self.window_size
        )  # split the dataset chuncks in sets with length of window_size
        Y = observation[1]  # get the label from the chunck
        nr_windows = len(chuncks)  # get nr of windows
        random_window = random.randint(
            0, nr_windows - 1
        )  # get random number between 0 and length of windows
        window = chuncks[random_window]
        if (
            len(window) != self.window_size
        ):  # if the window size is not equal to window size, get new one
            window = chuncks[random_window - 1]
            return window, Y
        else:
            return window, Y

    def batchloop(self) -> Tuple[List, List]:
        X = []  # noqa N806
        Y = []  # noqa N806
        for _ in range(self.batchsize):
            x, y = self.get_chuck()
            X.append(x)
            Y.append(y)
            self.index += 1
        return X, Y


class PaddedDataIterator(BaseDataIterator):
    """Iterator with additional padding of X

    Args:
        BaseDataIterator (_type_): _description_
    """

    def __init__(
        self,
        dataset: BaseDataset,
        window_size: int,
        batchsize: int,
        min_nr_lines: int = 1,
    ) -> None:
        self.dataset = dataset  # set dataset
        self.batchsize = batchsize  # set batchsize
        self.window_size = window_size  # set window size
        self.min_nr_lines = min_nr_lines

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.index_list) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
            return X_, torch.Tensor(Y)
        else:
            raise StopIteration

    def get_chuck(self) -> Tuple[List, int]:
        i = random.randint(
            0, len(self.dataset) - 1
        )  # get random number between 0 and length of chuncks
        observation = self.dataset.__getitem__(i)  # get the item from dataset
        chuncks = torch.split(
            observation[0], self.window_size
        )  # split the dataset chuncks in sets with length of window_size
        Y = observation[1]  # get the label from the chunck
        nr_windows = len(chuncks)  # get nr of windows
        random_window = random.randint(
            0, nr_windows - 1
        )  # get random number between 0 and length of windows
        window = chuncks[random_window]
        if (
            len(window) < self.min_nr_lines
        ):  # if the window size is smaller than min nr of lines in chuck
            window = chuncks[random_window - 1]  # get the previous chunck
            return window, Y
        else:
            return window, Y
