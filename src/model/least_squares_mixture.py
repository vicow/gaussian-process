from __future__ import print_function

import numpy as np
import scipy.linalg as lin

from .utils import ProgressBar
from .model import Model, ModelError
from collections import Counter


class LeastSquaresMixture(Model):

    # List of allowed keys for use with kwargs
    allowed_keys = ['K', 'beta', 'lam', 'iterations', 'epsilon', 'random_restarts']

    def __init__(self, X, y, K=2, beta=1, lam=0, iterations=1000, epsilon=1e-4, random_restarts=None, verbose=False):
        """
        :param K:           Number of mixture components
        :param beta:        Precision term for the probability of the data under the regression function
        :param lam:         Regularization parameter for the regression weights
        :param iterations:  Maximum number of iterations
        :param epsilon:     Condition for convergence
        :param verbose:     Display some information in the console
        """
        super(LeastSquaresMixture, self).__init__(self.__class__.__name__ + " (%s components)" % K, X, y)

        # Model hyperparameters
        self.K = K
        self.lam = lam
        self.iterations = iterations
        self.epsilon = epsilon

        # Model parameters
        N, D = X.shape
        self.w = np.zeros((D, K))
        self.pi = np.zeros(K)
        self.gamma = np.zeros((N, K))
        self.beta = beta
        self.marginal_likelihood = - np.inf
        self.trained = False
        self.random_restarts = random_restarts

    def __str__(self):
        description = "Model: %s\n" % self.name
        if self.trained:
            description += "Likelihood: %s\n" % self.marginal_likelihood
            description += "Beta: %s\n" % self.beta
            description += "Lambda: %s\n" % self.lam
            description += "Pi: %s\n" % self.pi
            description += "Weights: (norm: %s)\n" % [np.linalg.norm(self.w[:, i]) for i in range(self.K)]
            if self.w.shape[0] <= 20:
                description += "%s" % self.w
        else:
            description += "Not trained yet"
        return description

    def _expectation_maximization(self, verbose=False):
        """
        Learn the parameters for a mixture of least squares.
        Source:
        - http://stats.stackexchange.com/questions/33078/data-has-two-trends-how-to-extract-independent-trendlines/34287

        :return: Weights vectors, pi_k's, gamma's, beta and marginal likelihood
        """

        # Get the dimensions
        N = self.X.shape[0]
        D = self.X.shape[1] + 1  # + 1 to take bias into account

        if verbose:
            print("* Starting EM algorithm for mixture of K=%s least squares models" % self.K)
            print("* Beta = %s" % self.beta)
            print("* Lambda = %s" % self.lam)
            print("* Running at most %s iterations" % self.iterations)
            print("* Stopping when complete likelihood improves less than %s" % self.epsilon)

        # Add one's in order to find w0 (the bias)
        tX = np.concatenate((np.ones((N, 1)), self.X), axis=1)

        # Mixture weights
        pi = np.zeros(self.K) + .5

        # Expected mixture weights for each data point (responsibilities)
        gamma = np.zeros((N, self.K)) + .5

        # Regression weights
        w = np.random.rand(D, self.K)

        # Precision parameter
        beta = self.beta

        # Initialize likelihood
        complete_log_likelihood = - np.inf
        complete_log_likelihood_old = - np.inf

        if verbose:
            print("Obj\t\tpi1\t\tpi2\t\tw11\t\tw12\t\tw21\t\tw22\t\tbeta")

        for i in range(self.iterations):

            #### E-step

            # Compute Likelihood for each data point
            err = (np.tile(self.y, (1, self.K)) - np.dot(tX, w)) ** 2              # y - <w_k, x_n>
            prbs = - 0.5 * beta * err
            probabilities = 1 / np.sqrt(2 * np.pi) * np.sqrt(beta) * np.exp(prbs)  # N(y_n | <w_k, x_n>, beta^{-1})

            # Compute expected mixture weights
            gamma = np.tile(pi, (N, 1)) * probabilities
            gamma /= np.tile(np.sum(gamma, 1), (self.K, 1)).T

            # print(np.sum(gamma, 0))

            #### M-step

            # Max with respect to the mixture probabilities
            pi = np.mean(gamma, axis=0)

            # Max with respect to the regression weights
            for k in range(self.K):
                R_k = np.diag(gamma[:, k])
                R_kX = R_k.dot(tX)
                L = R_kX.T.dot(tX) + np.eye(D) * self.lam  # also try: lam / beta
                R = R_kX.T.dot(self.y)
                w[:, k] = lin.solve(L, R)[:, 0]

            # Max with respect to the precision term
            beta = float(N / np.sum(gamma * err))

            # Evaluate the complete data log-likelihood to test for convergence
            complete_log_likelihood = float(np.sum(np.log(np.sum(np.tile(pi, (N, 1)) * probabilities, axis=1))))

            if verbose:
                print("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f" % (complete_log_likelihood,
                                                                                  pi[0], pi[1],
                                                                                  w[0, 0], w[1, 0],
                                                                                  w[0, 1], w[1, 1],
                                                                                  beta))
            try:
                if np.isnan(complete_log_likelihood) \
                        or np.abs(complete_log_likelihood - complete_log_likelihood_old) < self.epsilon:
                    return w, pi, gamma, beta, complete_log_likelihood
            except:
                pass

            complete_log_likelihood_old = complete_log_likelihood

        # print("Hitting maximum iteration (%s)" % self.iterations)
        return w, pi, gamma, beta, complete_log_likelihood

    def train(self, seed=None, verbose=False, silent=False, **kwargs):
        """
        Train a mixture of least squares.

        :param seed:    Set the seed to fix the randomness
        :param verbose: Display details during expectation maximization
        :param silent:  Force to display nothing (a progress bar is displayed with random restarts even if non verbose)
        :param kwargs:  Set the hyperparameters if needed
        """
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.allowed_keys)
        if seed:
            np.random.seed(seed)
        if self.random_restarts and self.random_restarts > 0:
            w = pi = gamma = b = None
            marginal_likelihood = - np.inf
            if not verbose:
                if not silent:
                    bar = ProgressBar(self.random_restarts, count=True, text="Random restarts")
                    bar.start()
            for r in range(self.random_restarts):
                w_new, pi_new, gamma_new, b_new, marginal_likelihood_new = self._expectation_maximization(verbose=verbose)
                if marginal_likelihood_new > self.marginal_likelihood:
                    # print("Improved solution!")
                    w = w_new
                    pi = pi_new
                    gamma = gamma_new
                    b = b_new
                    marginal_likelihood = marginal_likelihood_new
                if not verbose:
                    if not silent:
                        bar.update(r)
            self.w = w
            self.pi = pi
            self.gamma = gamma
            self.beta = b
            self.marginal_likelihood = marginal_likelihood
        else:
            self.w, self.pi, self.gamma, self.beta, self.marginal_likelihood = self._expectation_maximization(verbose=verbose)
        self.trained = True

    def _compute_euclidean_distances(self, x):
        distances = []
        for x_i in self.X:
            distances.append(np.linalg.norm(x_i - x))
        return distances

    def _get_closest_point(self, x_new):
        distances = self._compute_euclidean_distances(x_new)
        return np.argmin(distances)

    def predict(self, x_new):
        if self.trained:
            x_new = list(x_new)
            if len(x_new) == self.X.shape[1]:
                n = self._get_closest_point(x_new)
                k = np.argmax(self.gamma[n, :])
                w_k = self.w[:, k]
                tx = np.ones((1, len(x_new )+ 1))
                tx[0, 1:] = x_new
                # print("Candidates: %s" % np.dot(tx, self.w))
                # print("Gamma     : %s" % self.gamma[n, :])
                # print("Chosen    : %s" % k)
                y_new = np.dot(tx, w_k)[0]
                return y_new, k
            else:
                raise(ModelError("Invalid size for new data point (%s instead of %s)" % (len(x_new), self.X.shape[1])))
        else:
            raise(ModelError("Model not trained"))

    def evaluate(self, X_test, y_test, verbose=False):
        if verbose:
            print("Evaluating model %s..." % self.name)
        se = 0
        accurate = 0
        chosen = Counter()
        if verbose:
            bar = ProgressBar(end_value=X_test.shape[0], text="Data point", count=True)
            bar.start()
        for i, x_new in enumerate(X_test):
            y_actual = y_test[i][0]
            y_new, k = self.predict(list(x_new))
            # print("Predicted: %s | Actual: %s" % (y_new, y_actual))
            chosen.update([k])
            se += (y_actual - y_new)**2
            if (y_new >= 1 and y_actual >= 1) or (y_new < 1 and y_actual < 1):
                accurate += 1
            if verbose:
                bar.update(i)

        rmse = np.sqrt(np.mean(se))
        accuracy = accurate / float(y_test.size)

        if verbose:
            print("Accuracy: %s" % accuracy)
            print("RMSE    : %s" % rmse)
            print("Chosen  : %s" % chosen)

        return rmse, accuracy, chosen
