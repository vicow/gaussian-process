# coding=utf-8
import os
import sys
import matplotlib
import numpy as np
import scipy as sp
import GPy
import pickle as cp
import matplotlib.pyplot as plt
from math import floor
from dataset import Sidekick

DATA_DIR = "../data/sidekick"

def predict_total_pledged(project, t, samples, m_s_test, m_f_test, pi, X, y):
    """Predict the total pledged money for a given project using a mixture of two GPs:

        p(Y_T | y_{1:t}, theta) = pi * p(y_T | y{1:t}, theta_s) + (1-pi) * p(y_T | y{1:t}, theta_f)

        where pi is the proportion of successful projects. Returns also the classification.
    """
    money = np.expand_dims(project.money[samples], axis=0)
    X_observed = np.ndarray(shape=(1, t), buffer=money, dtype=float)

    l_s = m_s_test.rbf.lengthscale
    sigma_f_s = m_s_test.rbf.variance
    sigma_n_s = m_s_test.Gaussian_noise.variance
    mean_s, var_s, likelihood_s, lml_s_pred = gaussian_process_regression(X, y, X_observed, k,
                                                              l=l_s,
                                                              sigma_n=sigma_n_s,
                                                              sigma_f=sigma_f_s)
    l_f = m_f_test.rbf.lengthscale
    sigma_f_f = m_f_test.rbf.variance
    sigma_n_f = m_f_test.Gaussian_noise.variance
    mean_f, var_f, likelihood_f, lml_f_pred = gaussian_process_regression(X, y, X_observed, k,
                                                              l=l_f,
                                                              sigma_n=sigma_n_f,
                                                              sigma_f=sigma_f_f)
    print(mean_s, var_s, likelihood_s)
    print(mean_f, var_f, likelihood_f)

    return likelihood_s > likelihood_f

def subsample(t0, t1, n_samples):
    t = t1 - t0
    if n_samples >= t:
        return range(t0, t1)
    samples = range(t0, t1, int(np.ceil(t / float(n_samples))))
    return samples


def k(xp, xq , l, sigma_f):
    """Covariance functions with squared exponential of length-scale l and signal noise sigma_f."""
    #return sigma_f * np.exp(-0.5 * np.linalg.norm(xp - xq) / float(l**2))
    return sigma_f * xp * xq


def K(x1, x2, l=1.0, sigma_f=1.0):
    """Compute the covariance matrix from the observations x."""
    cov_matrix = np.zeros((len(x1), len(x2)))
    for i, p in enumerate(x1):
        for j, q in enumerate(x2):
            cov_matrix[i, j] = k(p, q, l, sigma_f)
    return cov_matrix


def gaussian_process_regression(X, y, x_test, l=1.0, sigma_n=0.0, sigma_f=1.0):
    """
    Computes a regression using Gaussian Process using observations x and y = f(x) and a covariance function k.

    :param X        Inputs (NxD)
    :param y        Values at indices x (= f(x))
    :param x_test   Indices to get predicted values
    :param k        Covariance function
    :param sigma_n  Observations noise
    :return:        Mean m, variance var and log marginal likelihood lml
    """
    n = len(y)
    n_test = len(x_test)
    # Cholesky decompostion
    L = np.linalg.cholesky(K(X, X, l, sigma_f) + sigma_n * np.eye(n))
    a = sp.linalg.solve_triangular(L.T, sp.linalg.solve_triangular(L, y, lower=True))
    k_star = K(X, x_test, l, sigma_f)
    # Predictive mean
    m = np.dot(k_star.T, a)
    # Predictive variance
    v = sp.linalg.solve_triangular(L, k_star, lower=True)
    var = K(x_test, x_test, l, sigma_f) - v.T.dot(v)
    # Log maginal likelihood (last term is log2π / 2)
    lml_model = - 0.5 * np.sum(y * a) - np.sum(np.log(np.diag(L))) - n * 0.918938533205

    # Compute the likelihood of predicted values under the given model
    n_test = len(m)
    y_test = m
    L = np.linalg.cholesky(K(x_test, x_test, l, sigma_f) + sigma_n * np.eye(n_test))
    a = sp.linalg.solve_triangular(L.T, sp.linalg.solve_triangular(L, y_test, lower=True))
    lml_prediction = - 0.5 * np.sum(y_test * a) - np.sum(np.log(np.diag(L))) - n_test * 0.918938533205

    return m, var + sigma_n * np.eye(n_test), lml_model, lml_prediction

if __name__ == '__main__':
    X = np.expand_dims([1, 2, 3], axis=0).T
    y = np.expand_dims([1, 2, 3], axis=0).T
    x_test = np.expand_dims([4, 5], axis=0).T

    kernel = GPy.kern.Linear(input_dim=1, ARD=False)
    model = GPy.models.GPRegression(X, y, kernel)
    model.Gaussian_noise.variance = 0.01
    model.linear.variances = 1
    #model.optimize()
    #print model

    print("Mean\t\tVariance\tLog-likelihood\tLog-llh prediction")
    m, v, l, l_pred = gaussian_process_regression(X, y, x_test, sigma_n=0.01, sigma_f=1)
    print("%0.6f\t%0.6f\t%0.6f\t%0.6f" % (m[0], v[0][0], l, l_pred))
    m, v, l, l_pred = gaussian_process_regression(X, y, x_test, sigma_n=1, sigma_f=5)
    print("%0.6f\t%0.6f\t%0.6f\t%0.6f" % (m[0], v[0][0], l, l_pred))
