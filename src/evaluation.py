from dataset import Sidekick
import numpy as np
import GPy


def train(X, Y, kernel=None):
    """
    Train a GP on X inputs and Y independent outputs.

    :param X:   Input vector
    :param Y:   Independent outputs
    :return:    A trained (optimized) GP model
    """
    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=1)
    m = GPy.models.GPRegression(X, Y, kernel)
    m.optimize_restarts(num_restarts=10)
    return m

def evaluation():
    sk = Sidekick()
    sk.load()
    projects = sk.choose_n_projects()
    X = np.ndarray(shape=(1000, 1), buffer=np.arange(1000), dtype=int)
    Y = np.array([[status[1] for status in project['status']] for project in projects]).transpose()
    assert X.shape[0] == Y.shape[0]



if __name__ == '__main__':
    evaluation()