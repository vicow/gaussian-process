from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.abspath('../utils/'))  # Add sibling to Python path
sys.path.insert(0, os.path.abspath('../src/'))  # Add sibling to Python path
import matplotlib
matplotlib.rcParams['figure.figsize'] = (18,8)
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 16
import numpy as np
import pickle as cp
import matplotlib.pyplot as plt
from src.dataset import Sidekick
from src.model import LeastSquaresMixture
from math import floor

data_dir = "../../data/sidekick"

def subsample(t0, t1, n_samples):
    t = t1 - t0
    if n_samples >= t:
        return range(t0, t1)
    samples = range(t0, t1, int(np.ceil(t / float(n_samples))))
    return samples

sk = Sidekick(data_dir=data_dir)
sk.load()
projects_train, projects_test = sk.split()

N = 1000
N_train = int(floor(0.8*N))
seed = 2
t0 = 1
t1 = 500
n_samples = 25
T = 999

samples = subsample(t0, t1, n_samples)
t = len(samples)

#N_projects = sk.choose_n_projects(n=N, seed=seed)
#projects_train = N_projects[:N_train]
#projects_test = N_projects[N_train:]

projects_train = [p for p in projects_train if p.money[T] < 100]

X_train = np.ndarray(shape=(len(projects_train), t), buffer=np.array([p.money[samples] for p in projects_train]), dtype=float)
y_train = np.expand_dims(np.array([p.money[T] for p in projects_train]), axis=1)
X_test = np.ndarray(shape=(len(projects_test), t), buffer=np.array([p.money[samples] for p in projects_test]), dtype=float)
y_test = np.expand_dims(np.array([p.money[T] for p in projects_test]), axis=1)

# Required to contain the prediction in a reasonable range
# The problem arises when evaluating the likelihood in the expression for gamma_nk
X_max = np.max(X_train, axis=0)
X_train = X_train / X_max[np.newaxis, :]
# Apply same preprocessing to testing set
X_test = X_test / X_max[np.newaxis, :]

#y_max = np.max(y_train, axis=0)
#y_train = y_train / y_max[np.newaxis, :]
#y_test = y_test / y_max[np.newaxis, :]


print("Training on %s projects" % len(X_train))
print("Testing on %s projects" % len(X_test))
print("Number of features: %s" % n_samples)

K = 3
beta = 1 / np.var(y_train)
epsilon = 1e-2
lam = 0.01
iterations = 1000
random_restarts = 1

mls = LeastSquaresMixture(X_train, y_train,
                          K=K, beta=beta, lam=lam,
                          iterations=iterations, epsilon=epsilon, random_restarts=random_restarts)
mls.train(verbose=True)

print(mls)

rmse, accuracy, chosen = mls.evaluate(X_test, y_test, verbose=True)