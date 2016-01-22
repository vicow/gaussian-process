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
from utils.progressbar import ProgressBar

data_dir = "../../data/sidekick"

def subsample(t0, t1, n_samples):
    t = t1 - t0
    if n_samples >= t:
        return range(t0, t1)
    samples = range(t0, t1, int(np.ceil(t / float(n_samples))))
    return samples

seed = 2
N = 1000
N_train = int(floor(0.8*N))

#N_projects = sk.choose_n_projects(n=N, seed=seed)
#projects_train = N_projects[:N_train]
#projects_test = N_projects[N_train:]

# Required to contain the prediction in a reasonable range
# The problem arises when evaluating the likelihood in the expression for gamma_nk
#X_max = np.max(X_train, axis=0)
#X_train = X_train / X_max[np.newaxis, :]
# Apply same preprocessing to testing set
#X_test = X_test / X_max[np.newaxis, :]

#y_max = np.max(y_train, axis=0)
#y_train = y_train / y_max[np.newaxis, :]
#y_test = y_test / y_max[np.newaxis, :]


#print("Training on %s projects" % len(X_train))
#print("Testing on %s projects" % len(X_test))
#print("Number of features: %s" % n_samples)

def one_run(projects_train, projects_test):
    rmse_run = []
    accuracy_run = []
    relative_time = np.linspace(0.025, 1, 5)
    bar = ProgressBar(end_value=len(relative_time), text="Observed values", count=True)
    bar.start()
    for i, rel_t in enumerate(relative_time):
        # Data
        #t0 = 1
        #n_samples = 1
        #samples = subsample(t0, t1, n_samples)
        #t = len(samples)
        samples = rel_t * 1000 - 1
        t = 1
        T = 999

        # Remove outliers
        projects_train = [p for p in projects_train if p.money[T] < 10]

        X_train = np.ndarray(shape=(len(projects_train), t), buffer=np.array([p.money[samples] for p in projects_train]), dtype=float)
        y_train = np.expand_dims(np.array([p.money[T] for p in projects_train]), axis=1)
        X_test = np.ndarray(shape=(len(projects_test), t), buffer=np.array([p.money[samples] for p in projects_test]), dtype=float)
        y_test = np.expand_dims(np.array([p.money[T] for p in projects_test]), axis=1)

        # Hyperparameters
        K = 2
        beta = 1  # 1 / np.var(y_train)
        epsilon = 1e0
        lam = 0
        iterations = 1000
        random_restarts = None

        mls = LeastSquaresMixture(X_train, y_train,
                                  K=K, beta=beta, lam=lam,
                                  iterations=iterations, epsilon=epsilon, random_restarts=random_restarts)
        mls.train(verbose=False)
        #print(mls)

        rmse, accuracy = mls.evaluate(X_test, y_test, verbose=False)
        rmse_run.append(rmse)
        accuracy_run.append(accuracy)

        bar.update(i)

    return rmse_run, accuracy_run


def learning_curve(seed=2):
    sk = Sidekick(data_dir=data_dir, seed=seed)
    sk.load(light=False)

    rmse_all = []
    accuracy_all = []
    for r in range(10):
        projects_train, projects_test = sk.split(threshold=0.7, shuffle=True)
        n_test = len(projects_test)
        projects_validation = projects_test[:floor(n_test*3/5)]
        projects_test = projects_test[floor(n_test*3/5):]

        rmse_run, accuracy_run = one_run(projects_train, projects_test)
        with open('rmse_run.pkl', 'wb') as f:
            cp.dump(rmse_all, f)
        with open('accuracy_run.pkl', 'wb') as f:
            cp.dump(accuracy_all, f)

        rmse_all.append(rmse_run)
        accuracy_all.append(accuracy_run)
        with open('rmse_all.pkl', 'wb') as f:
            cp.dump(rmse_all, f)
        with open('accuracy_all.pkl', 'wb') as f:
            cp.dump(accuracy_all, f)

    return rmse_all, accuracy_all


rmse_all, accuracy_all = learning_curve()
with open('rmse_all.pkl', 'wb') as f:
    cp.dump(rmse_all, f)
with open('accuracy_all.pkl', 'wb') as f:
    cp.dump(accuracy_all, f)
