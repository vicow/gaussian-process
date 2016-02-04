from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.abspath('../utils/'))  # Add sibling to Python path
sys.path.insert(0, os.path.abspath('../src/'))  # Add sibling to Python path
import numpy as np
import pickle as cp
from src.dataset import Sidekick
from src.model import LeastSquaresMixture
from math import floor
from utils import ProgressBar

data_dir = "../../data/sidekick"

def subsample(t0, t1, n_samples):
    t = t1 - t0
    if n_samples >= t:
        return range(t0, t1)
    samples = range(t0, t1, int(np.ceil(t / float(n_samples))))
    return samples

def one_run(projects_train, projects_test, outlier_threshold):
    rmse_failed_run = []
    rmse_success_run = []
    rmse_run = []
    accuracy_run = []
    relative_time = np.linspace(0.025, 1, 40)
    bar = ProgressBar(end_value=len(relative_time), text="Time steps", count=True)
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
        projects_train = [p for p in projects_train if p.money[T] < outlier_threshold and p.money[samples] < outlier_threshold]
        projects_test = [p for p in projects_test if p.money[T] < outlier_threshold and p.money[samples] < outlier_threshold]

        X_train = np.ndarray(shape=(len(projects_train), t), buffer=np.array([p.money[samples] for p in projects_train]), dtype=float)
        y_train = np.expand_dims(np.array([p.money[T] for p in projects_train]), axis=1)
        X_test = np.ndarray(shape=(len(projects_test), t), buffer=np.array([p.money[samples] for p in projects_test]), dtype=float)
        y_test = np.expand_dims(np.array([p.money[T] for p in projects_test]), axis=1)

        #X_max = np.max(X_train, axis=0)
        #X_train = X_train / X_max[np.newaxis, :]
        #X_test = X_test / X_max[np.newaxis, :]

        # Hyperparameters
        K = 2
        beta = 1 / np.var(y_train)
        epsilon = 1e0
        lam = 0
        iterations = 1000
        random_restarts = None

        mls = LeastSquaresMixture(X_train, y_train,
                                  K=K, beta=beta, lam=lam,
                                  iterations=iterations, epsilon=epsilon, random_restarts=random_restarts)
        mls.train(verbose=False)
        #print(mls)

        rmse_failed, rmse_success, rmse, accuracy = mls.evaluate(X_test, y_test, verbose=False)
        rmse_failed_run.append(rmse_failed)
        rmse_success_run.append(rmse_success)
        rmse_run.append(rmse)
        accuracy_run.append(accuracy)

        bar.update(i)

    return rmse_failed_run, rmse_success_run, rmse_run, accuracy_run


def learning_curve(seed=2, runs=10, light=False, outlier_threshold=10):
    sk = Sidekick(data_dir=data_dir, seed=seed)
    sk.load(light=light)

    rmse_failed_all = []
    rmse_success_all = []
    rmse_all = []
    accuracy_all = []
    for r in range(runs):
        projects_train, projects_test = sk.split(threshold=0.7, shuffle=True)
        n_test = len(projects_test)
        projects_validation = projects_test[:floor(n_test*3/5)]
        projects_test = projects_test[floor(n_test*3/5):]

        rmse_failed_run, rmse_success_run, rmse_run, accuracy_run = one_run(projects_train, projects_test, outlier_threshold)
        rmse_failed_all.append(rmse_failed_run)
        rmse_success_all.append(rmse_success_run)
        rmse_all.append(rmse_run)
        accuracy_all.append(accuracy_run)
        with open('rmse_failed_outlier_%s.pkl' % outlier_threshold, 'wb') as f:
            cp.dump(rmse_failed_all, f)
        with open('rmse_success_outlier_%s.pkl' % outlier_threshold, 'wb') as f:
            cp.dump(rmse_success_all, f)
        with open('rmse_outlier_%s.pkl' % outlier_threshold, 'wb') as f:
            cp.dump(rmse_all, f)
        with open('accuracy_outlier_%s.pkl' % outlier_threshold, 'wb') as f:
            cp.dump(accuracy_all, f)

    return rmse_failed_all, rmse_success_all, rmse_all, accuracy_all


rmse_failed_all, rmse_success_all, rmse_all, accuracy_all = learning_curve(seed=2,
                                                                           runs=10,
                                                                           light=False,
                                                                           outlier_threshold=2)
