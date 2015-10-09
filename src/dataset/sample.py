from abc import ABCMeta, abstractmethod
try:
    import cPickle as cp  # Python 2
except ImportError:
    import pickle as cp  # Python 3
import numpy as np


class SampleABC:
    """
    Abstract class representing a sample from a dataset.

    :param  data_dir:    Base data directory
    :type   data_dir:    str
    """
    __metaclass__ = ABCMeta

    data_dir = "../data"


class Sample(SampleABC):
    """
    Mother class for a sample with utility methods.
    """

    def __init__(self): pass

    @staticmethod
    def _load_binary(file_name):
        """
        Load binary file with cPickle.

        :param file_name:   Name of file
        :return:            Data in file.
        """
        with open(file_name, 'rb') as f:
            return cp.load(f)

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
    def _split(x, threshold=0.8):
        """
        Split a series x in training and testing set.

        :param x:           Series to split
        :param threshold:   Threshold to split
        :return:            Train set, test set
        """
        train_size = int(np.floor(len(x) * threshold))
        x_train = np.array(x[:train_size])
        x_test = np.array(x[train_size:])

        return x_train, x_test
