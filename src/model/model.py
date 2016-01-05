from abc import ABCMeta, abstractmethod
import numpy as np
try:
    import cPickle as cp  # Python 2
except ImportError:
    import pickle as cp  # Python 3


class ModelABC:
    """
    Abstract class representing a model.

    """
    __metaclass__ = ABCMeta



class Model(ModelABC):
    """
    Mother class for a model with utility methods.
    """

    def __init__(self, name, X, y):
        self.name = name
        if np.ndim(X) != 2:
            raise(ModelError("Expecting ndarray for X"))
        else:
            self.X = X
        if np.ndim(y) != 2:
            raise ModelError("Expecting ndarray for y")
        else:
            self.y = y


class ModelError(Exception):
    def __init__(self, message):
        self.message = message

