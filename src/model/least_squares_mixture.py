from __future__ import print_function
from .model import Model, ModelError
import numpy as np
import scipy.linalg as lin


class LeastSquaresMixture(Model):
    def __init__(self, X, y):
        super(LeastSquaresMixture, self).__init__(self.__class__.__name__, X, y)
        self.w = None
        self.pi = None
        self.beta = None
        self.K = None

    def _expectation_maximization(self, K=2, beta=100, lam=0.01, iterations=100, epsilon=1e-4, verbose=False):
        """
        Compute the weights for a mixture of least squares.
        Source:
        - http://stats.stackexchange.com/questions/33078/data-has-two-trends-how-to-extract-independent-trendlines/34287

        :param K:           Number of mixture components
        :param beta:        Precision term for the probability of the data under the regression function
        :param lam:         Regularization parameter for the regression weights
        :param iterations:  Maximum number of iterations
        :param epsilon:     Condition for convergence
        :param verbose:     Display some information in the console
        :return:            Weights vectors for each line, pi_k's and beta parameter
        """
        N = self.X.shape[0]
        D = self.X.shape[1]

        if verbose:
            print("* Starting EM algorithm for mixture of K=%s least squares models" % K)
            print("* Beta = %s" % beta)
            print("* Lambda = %s" % lam)
            print("* Running at most %s iterations" % iterations)
            print("* Stopping when complete likelihood improves less than %s" % epsilon)

        # Transform to ndarray
        self.y = np.expand_dims(self.y, axis=1)

        # Add one's in order to find w0 (the bias)
        tX = np.concatenate((np.ones((N, 1)), self.X), axis=1)

        # Mixture weights
        pi = np.zeros(K) + .5

        # Expected mixture weights for each data point (responsibilities)
        gamma = np.zeros((N, K)) + .5

        # Regression weights
        w = np.random.rand(D, K)

        if verbose:
            print("Obj\t\tpi1\t\tpi2\t\tw11\t\tw12\t\tw21\t\tw22\t\tbeta")

        for _ in xrange(iterations):
            # if 0:
            #     plt.plot(r, np.dot(rx, w1), '-r', alpha=.5)
            #     plt.plot(r, np.dot(rx, w2), '-g', alpha=.5)

            #### E-step

            # Compute Likelihood for each data point
            err = (np.tile(self.y, (1, K)) - np.dot(tX, w)) ** 2                               # y - <w_k, x_n>
            prbs = - 0.5 * beta * err
            probabilities = 1 / np.sqrt(2 * np.pi) * np.sqrt(beta) * np.exp(prbs)      # N(y_n | <w_k, x_n>, beta^{-1})

            # Compute expected mixture weights
            gamma = np.tile(pi, (N, 1)) * probabilities
            gamma /= np.tile(np.sum(gamma, 1), (K, 1)).T

            #### M-step

            # Max with respect to the mixture probabilities
            pi = np.mean(gamma, axis=0)

            # Max with respect to the regression weights
            for k in xrange(K):
                R_k = np.diag(gamma[:, k])
                R_kX = R_k.dot(tX)
                L = R_kX.T.dot(tX) + np.eye(2) * lam / beta
                R = R_kX.T.dot(self.y)
                w[:, k] = lin.solve(L, R)[:, 0]

            # Max with respect to the precision term
            beta = float(N / np.sum(gamma * err))

            # Evaluate the complete data log-likelihood to test for convergence
            complete_log_likelihood = np.sum(np.log(np.sum(np.tile(self.pi, (N, 1)) * probabilities, axis=1)))

            if verbose:
                print("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f" % (complete_log_likelihood[0],
                                                                                  pi[0], pi[1],
                                                                                  w[0, 0], w[1, 0],
                                                                                  w[0, 1], w[1, 1],
                                                                                  beta))
            try:
                if np.isnan(complete_log_likelihood): break
                if np.abs(complete_log_likelihood - complete_log_likelihood_old) < epsilon: break
            except:
                pass

            complete_log_likelihood_old = complete_log_likelihood

        return w, pi, beta

    def posterior(self, x_new, y_new, k):
        tx = np.array([1, x_new])
        w_k = self.w[:, k]
        pi_k = self.pi[k]
        return - pi_k * 0.5 * (np.log(2 * np.pi) - np.log(self.beta) + (y_new - np.dot(w_k, tx))**2 * self.beta)

    def train(self, K=2, beta=100, lam=0.01, iterations=100, epsilon=1e-4, verbose=False):
        self.w, self.pi, self.beta = self._expectation_maximization(K, beta, lam, iterations, epsilon, verbose)

    def predict(self, x_new):
        tx = np.array([1, x_new])
        y_new = np.dot(tx, self.w)
        posteriors = []
        return y_new, posteriors
