from __future__ import print_function
# sys.path.insert(0, os.path.abspath('../utils/'))  # Add sibling to Python path
# sys.path.insert(0, os.path.abspath('../src/'))  # Add sibling to Python path
import time
import numpy as np

from src.dataset import Sidekick
from utils import ProgressBar, Utils
from sklearn import linear_model
import argparse

data_dir = "../../data/sidekick"
u = Utils()


def _average_increment(money):
    """
    Compute the average increment.

    :param money:   Time series
    :return:        Average increment
    """
    if len(money) > 0:
        d = np.diff(money)
        increments = [i for i in d if i != 0]
        if len(increments) > 0:
            return np.mean(increments)
        else:
            return 0
    else:
        return 0


def _max_increment(money):
    if len(money) > 0:
        d = np.diff(money)
        if len(d) == 0:
            return 0
        else:
            return np.max(np.diff(money))
    else:
        return 0


def _get_extractor(features):
    """
    Return the corresponding feature extractor.

    :param features:    Key to select a feature extractor
    :return:            Function to extract features from an array
    """
    if features == "last-sample" or features is None:
        return lambda a: a[-1]
    elif features == "derivative":
        return lambda a: a[-1] / len(a)
    elif features == "average-increment":
        return lambda a: _average_increment(a)
    elif features == "max-increment":
        return lambda a: _max_increment(a)
    else:
        raise AttributeError("Undefined attribute %s, should be one of 'last-sample', 'derivative', 'average-increment', 'max-increment'" % features)


def subsample(t, granularity):
    """
    Compute an index set between 0 and t at a the given granularity. The granularity controls the density of samples.
    For instance, a granularity of 1.0 means 100% of the samples, a granularity of 0.5 means 50% of the samples and
    a granularity of 0.1 means 10% of the samples.

    :param t:           Upper bound of the index set
    :param granularity: Density of the samples
    :return:            A linear space between 0 and t with granularity * t samples
    """
    if granularity > 1.0 or granularity <= 0:
        raise ValueError("granularity must be in ]0, 1]")
    t0 = 0
    n_samples = int(np.ceil(granularity * t))
    if n_samples == 1:
        return [t]
    else:
        return np.linspace(t0, t, n_samples, dtype=int)


def _evaluate(X_test, y_test, projects_test, model, normalized):
    """
    Evaluate the accuracy and RMSE for a given test set under a given model.

    :param X_test:          Test features
    :param y_test:          Test targets
    :param projects_test:   Test projects set
    :param model:           Model used for predictions
    :param normalized:      Whether to the normalized money
    :return:                RMSE and accuracy (tuple)
    """
    se_total = []
    accuracy = 0
    for i, x_test in enumerate(X_test):
        if normalized:
            goal = 1
        else:
            p = projects_test[i]
            goal = p.goal
        x_test = np.expand_dims(x_test, axis=0)
        y_pred = model.predict(x_test)
        y_pred = y_pred[0]
        y_actual = y_test[i]
        se = (y_pred - y_actual) ** 2
        se_total.append(se)
        if (y_pred / goal >= 1 and y_actual / goal >= 1) or (y_pred / goal < 1 and y_actual / goal < 1):
            accuracy += 1

    rmse_total = np.sqrt(np.mean(se_total))
    accuracy /= len(y_test)

    return rmse_total, accuracy


def _one_run(projects_train, projects_test, relative_time, features, outlier_threshold, normalized, granularity):
    """
    Run the experiment once for an increasing time for some given parameters.

    :param projects_train:      Training projects set
    :param projects_test:       Test projects set
    :param realtive_time:       Relative time, used as x axis
    :param features:            Features to extract from money time series
    :param outlier_threshold:   Threshold to discard outliers
    :param normalized:          Whether to use normalized money
    :param granularity:         Level of granularity
    :return:                    RMSE and accuracy fot this experiment
    """
    rmse_run = []
    accuracy_run = []
    bar = ProgressBar(end_value=len(relative_time), text="Time steps", count=True)
    bar.start()

    # Remove outliers
    projects_train_filtered = [p for p in projects_train if np.all([(m - outlier_threshold) <= 0 for m in p.money])]
    projects_test_filtered = [p for p in projects_test if np.all([(m - outlier_threshold) <= 0 for m in p.money])]

    for i, rel_t in enumerate(relative_time):
        # Data
        t = int(rel_t * 999)
        samples = subsample(t, granularity)
        n_samples = 1
        T = 999

        X_train = np.ndarray(shape=(len(projects_train_filtered), n_samples),
                             buffer=np.array([features(p.money[samples]) for p in projects_train_filtered]),
                             dtype=float)
        y_train = np.expand_dims(np.array([p.money[T] for p in projects_train_filtered]), axis=1)
        X_test = np.ndarray(shape=(len(projects_test_filtered), n_samples),
                            buffer=np.array([features(p.money[samples]) for p in projects_test_filtered]),
                            dtype=float)
        y_test = np.expand_dims(np.array([p.money[T] for p in projects_test_filtered]), axis=1)

        m = linear_model.LinearRegression()
        m.fit(X_train, y_train)

        rmse, accuracy = _evaluate(X_test, y_test, projects_test, m, normalized)
        rmse_run.append(rmse)
        accuracy_run.append(accuracy)

        bar.update(i)

    return rmse_run, accuracy_run


def experiment(args):
    """
    Run the experiment for the given number of times.

    :param seed:                Seed to use when shuffling the data set
    :param runs:                Number of times to run the experiment
    :param light:               Whether to use a light data set (1000 projects)
    :param outlier_threshold:   Threshold of outliers to discard
    :param normalized:          Whether to use the normalized money
    :param granularity:         Level of granularity
    :return:
    """
    features = _get_extractor(args.features)

    sk = Sidekick(data_dir=data_dir, seed=args.seed)
    sk.load(light=args.light)

    relative_time = np.linspace(0.025, 1, 40)

    # Construct data dict
    data_rmse = {
        "plot_label": args.features,
        "x": relative_time,
        "y": [],
        "args": vars(args),
        "timestamp": time.time()
    }
    data_accuracy = {
        "plot_label": args.features,
        "x": relative_time,
        "y": [],
        "args": vars(args),
        "timestamp": time.time()
    }
    rmse_all = []
    accuracy_all = []
    for r in range(args.runs):
        projects_train, projects_test = sk.split(threshold=0.7, shuffle=True)

        # Set which money time series to use
        for p in np.append(projects_train, projects_test):
            p.normalized = args.normalized

        # n_test = len(projects_test)
        # projects_validation = projects_test[:floor(n_test*3/5)]
        # projects_test = projects_test[floor(n_test*3/5):]

        # Run the experiment once
        rmse_run, accuracy_run = _one_run(projects_train, projects_test, relative_time, features,
                                          args.outlierThreshold, args.normalized, args.granularity)

        # Record the results
        rmse_all.append(rmse_run)
        accuracy_all.append(accuracy_run)

    data_rmse["y"] = rmse_all
    data_accuracy["y"] = accuracy_all

    # Save the results to disk
    args.metric = "rmse"
    u.save_args(data_rmse, vars(args))
    args.metric = "accuracy"
    u.save_args(data_accuracy, vars(args))

    return rmse_all, accuracy_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', default="linear-regression", help="Label to identify the experiment")
    parser.add_argument('--seed', default=2, help="Seed to use when shuffling the data set")
    parser.add_argument('--runs', default=10, help="Number of times to run the experiment")
    parser.add_argument('--light', default=False, help="Whether to use a light data set (1000 projects)")
    parser.add_argument('--outlierThreshold', default=10000, help="Threshold of outliers to discard")
    parser.add_argument('--normalized', default=False, help="Whether to use the normalized money")
    parser.add_argument('--granularity', default=1.0, help="Level of granularity")
    parser.add_argument('--features', default="max-increment", help="Which features extractor to use")

    args = parser.parse_args()

    u.args_sanity_check(vars(args))

    rmse_all, accuracy_all = experiment(args)
