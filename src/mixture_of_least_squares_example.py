# Source: http://stats.stackexchange.com/questions/33078/data-has-two-trends-how-to-extract-independent-trendlines/34287

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin

#np.random.seed(1)

# Generate some random data
N = 100
X = np.random.rand(N, 2)
X[:, 0] = 1

w = np.random.rand(2, 2)
y = np.zeros(N)

n = int(np.random.rand(1, 1) * N)
y[:n] = np.dot(X[:n, :], w[0, :]) + np.random.normal(size=n) * .01
y[n:] = np.dot(X[n:, :], w[1, :]) + np.random.normal(size=N - n) * .01

rx = np.ones((100, 2))
r = np.arange(0, 1, .01)
rx[:, 1] = r

# Plot the random dataset
plt.plot(X[:, 1], y, '.b')
plt.plot(r, np.dot(rx, w[0, :]), ':k', linewidth=2)
plt.plot(r, np.dot(rx, w[1, :]), ':k', linewidth=2)


def em(X, y, K=2, beta=1.0, lam=0.01, iterations=100, epsilon=1e-4, verbose=False):
    """
    Compute the weights for a mixture of least squares.

    :param X:           Input variables (features)
    :param y:           Target vector
    :param K:           Number of mixture components
    :param beta:        Precision term for the probability of the data under the regression function
    :param lam:         Regularization parameter for the regression weights
    :param iterations:  Maximum number of iterations
    :param epsilon:     Condition for convergence
    :param verbose:     Display some information in the console
    :return:            Weights vectors for each line, pi_k's and beta parameter
    """
    N = X.shape[0]
    D = X.shape[1]

    if verbose:
        print "* Starting EM algorithm for mixture of K=%s least squares models" % K
        print "* Beta = %s" % beta
        print "* Lambda = %s" % lam
        print "* Running at most %s iterations" % iterations
        print "* Stopping when complete likelihood improves less than %s" % epsilon

    # Transform to ndarray
    y = np.expand_dims(y, axis=1)

    # Mixture weights
    pi = np.zeros(K) + .5

    # Expected mixture weights for each data point (responsibilities)
    gamma = np.zeros((N, K)) + .5

    # Regression weights
    w = np.random.rand(D, K)

    if verbose:
        print "Obj\t\tpi1\t\tpi2\t\tw11\t\tw12\t\tw21\t\tw22\t\tbeta"

    for _ in xrange(iterations):
        # if 0:
        #     plt.plot(r, np.dot(rx, w1), '-r', alpha=.5)
        #     plt.plot(r, np.dot(rx, w2), '-g', alpha=.5)

        #### E-step

        # Compute Likelihood for each data point
        err = (np.tile(y, (1, K)) - np.dot(X, w)) ** 2                                                # y - <w_k, x_n>
        prbs = - 0.5 * beta * err
        probabilities = 1 / np.sqrt(2 * np.pi) * np.sqrt(beta) * np.exp(prbs)           # N(y_n | <w_k, x_n>, beta^{-1})

        # Compute expected mixture weights
        gamma = np.tile(pi, (N, 1)) * probabilities
        gamma /= np.tile(np.sum(gamma, 1), (K, 1)).T

        #### M-step

        # Max with respect to the mixture probabilities
        pi = np.mean(gamma, axis=0)

        # Max with respect to the regression weights
        for k in xrange(K):
            R_k = np.diag(gamma[:, k])
            R_kX = R_k.dot(X)
            L = R_kX.T.dot(X) + np.eye(2) * lam / beta
            R = R_kX.T.dot(y)
            w[:, k] = lin.solve(L, R)[:, 0]

        # gamma1x = np.tile(gamma[:, 0], (2, 1)).T * X
        # xp1 = np.dot(gamma1x.T, X) + np.eye(2) * lam * beta
        # yp1 = np.dot(gamma1x.T, y)
        # w1 = lin.solve(xp1, yp1)

        # gamma2x = np.tile(gamma[:, 1], (2, 1)).T * X
        # xp2 = np.dot(gamma2x.T, X) + np.eye(2) * lam * beta
        # yp2 = np.dot(gamma[:, 1] * y, X)
        # w2 = lin.solve(xp2, yp2)

        # Max with respect to the precision term
        beta = float(N / np.sum(gamma * err))

        # Evaluate the complete data log-likelihood to test for convergence
        complete_log_likelihood = np.sum(np.log(np.sum(np.tile(pi, (N, 1)) * probabilities, axis=1)))

        if verbose:
            print "%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f" % (complete_log_likelihood, pi[0], pi[1], w[0, 0], w[1, 0], w[0, 1], w[1, 1], beta)

        try:
            if np.isnan(complete_log_likelihood): break
            if np.abs(complete_log_likelihood - complete_log_likelihood_old) < epsilon: break
        except:
            pass

        complete_log_likelihood_old = complete_log_likelihood

    return w, pi, beta


def log_likelihood(x, y, w_k, beta):
    tx = np.array([1, x])
    return - 0.5 * (np.log(2 * np.pi) - np.log(beta) + beta * (y - np.dot(w_k, tx))**2)


def posterior(x, y, w_k, pi_k, beta):
    likelihood = np.exp(log_likelihood(x, y, w_k, beta))
    return pi_k * likelihood


def normal_pdf(x, mu, sigma2):
    return 1 / np.sqrt(2*np.pi*sigma2) * np.exp(-0.5 * (x - mu) **2 / sigma2)


# run the em algorithm and plot the solution
w, pi, beta = em(X, y, beta=0.03, epsilon=1e-10, lam=0, verbose=True)
w1 = w[:, 0]
w2 = w[:, 1]
pi1 = pi[0]
pi2 = pi[1]
plt.plot(r, np.dot(rx, w1), '-r')
plt.plot(r, np.dot(rx, w2), '-g')

beta = 1
# New point to display
x_new = 0.9
#y_new = 0.8
tx_new = [1, x_new]
y_new1 = np.dot(tx_new, w1)
y_new2 = np.dot(tx_new, w2)
y_hat = pi1 * y_new1 + pi2 * y_new2

x_red = plt.plot(x_new, y_new1, 'rx', mew=4, ms=10)
x_green = plt.plot(x_new, y_new2, 'gx', mew=4, ms=10)
x_black = plt.plot(x_new, y_hat, 'kx', mew=4, ms=10)
#plt.plot(x_new, y_new, 'kx', mew=4, ms=10)

likelihood_red = log_likelihood(x_new, y_hat, w1, beta)
likelihood_green = log_likelihood(x_new, y_hat, w2, beta)

posterior_red = posterior(x_new, y_hat, w1, pi1, beta)
posterior_green = posterior(x_new, y_hat, w2, pi2, beta)
print posterior_green, posterior_red
normalization = posterior_red + posterior_green
print normalization
posterior_red /= normalization
posterior_green /= normalization
print posterior_red, posterior_green

print "More likely to be generated by %s line" % ("red" if likelihood_red > likelihood_green else "green")
print "New point at (%s, %s) is composed of %0.2f%% red and %0.2f%% green" % (x_new, y_hat, posterior_red, posterior_green)

plt.savefig('MLS.pdf')
