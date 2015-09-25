from abc import ABCMeta, abstractmethod
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
    def load(self): pass


class Dataset(DatasetABC):
    """
    Mother class for a dataset with utility methods.
    """

    def __init__(self, name):
        self.name = name

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

