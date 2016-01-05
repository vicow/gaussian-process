from abc import ABCMeta, abstractmethod
import numpy as np
try:
    import cPickle as cp  # Python 2
except ImportError:
    import pickle as cp  # Python 3


class DatasetABC:
    """
    Abstract class representing a dataset.

    :param  data_dir:    Base data directory
    :type   data_dir:    str
    """
    __metaclass__ = ABCMeta

    data_dir = "../data"

    @abstractmethod
    def load(self):
        """
        Load a dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, item):
        """
        Access a sample from the dataset.
        """
        pass

    def __iter__(self):
        """
        Iterate over samples in dataset.
        """
        pass


class Dataset(DatasetABC):
    """
    Mother class for a dataset with utility methods.
    """

    def __init__(self, name):
        self.name = name
        self.data = []

    def __getitem__(self, item):
        """
        Access a sample from the dataset.
        """
        return self.data[item]

    def __iter__(self):
        """
        Iterate over samples in dataset.
        """
        for sample in self.data:
            yield sample

    def split(self):
        return self._split(self.data)

    @staticmethod
    def _load_binary(file_name):
        """
        Load binary file with cPickle.

        :param file_name:   Name of file
        :return:            Data in file.
        """
        try:
            with open(file_name, 'rb') as f:
                return cp.load(f)
        except UnicodeDecodeError:  # When loading Python 2 pickle from Python 3
            with open(file_name, 'rb') as f:
                return cp.load(f, encoding="latin1")


    @staticmethod
    def _save_binary(file_name, data):
        """
        Save binary data to disk.

        :param file_name:   Name of file
        :param data:        Data to save
        """
        with open(file_name, "wb") as f:
            cp.dump(data, f)

    @staticmethod
    def _split(x, threshold=0.8, shuffle=False):
        """
        Split a series x in training and testing set.

        :param x:           Series to split
        :param threshold:   Threshold to split
        :param randomize:   Shuffle the series
        :return:            Train set, test set
        """
        x = np.array(x)
        if shuffle:
            x.shuffle()
        train_size = int(np.floor(len(x) * threshold))
        x_train = x[:train_size]
        x_test = x[train_size:]

        return x_train, x_test


class DatasetError(Exception):
    def __init__(self): pass

