import matplotlib.pyplot as plt
import pickle as cp
import numpy as np
import argparse
from utils import Utils, ArgumentError

u = Utils()


def _build_plot_description(metric):
    """
    Build the plot description based on the metric
    :param metric:  One of accuracy or rmse
    :return:        Title, x label and y label
    """
    if metric == "accuracy":
        title = "Median accuracy over 10 runs using linear regression"
        y_label = "Accuracy (%)"
    elif metric == "rmse":
        title = "Median RMSE over 10 runs using linear regression"
        y_label = "RMSE"
    else:
        raise ArgumentError("Metric argument '%s' invalid" % metric)
    x_label = "Relative time"

    return title, x_label, y_label


def plot_data(data, title, x_label, y_label, args):
    """
    Plot the data.

    :param data:        List of data dict
    :param title:       Title of plot
    :param x_label:     X label
    :param y_label:     Y label
    :param args:        Arguments of the script
    """
    for d in data:
        y = np.median(d["y"], axis=0) * 100 if "y" in d else d["y_value"]
        y_err = np.std(d["y"], axis=0) * 100 if "y" in d else d["y_err"]
        granularity = ""
        if "args" in d:
            granularity += " " + r'$\gamma = %s$' % d["args"]["granularity"]
        plt.errorbar(d["x"], y, y_err, label=d["plot_label"] + granularity)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid()
    if args.metric == "accuracy":
        plt.ylim([plt.ylim()[0], 100])
    fig_name = "fig/%s.pdf" % "-".join([args.label, args.metric, "-".join(args.folder.split("/")[1:])])
    plt.savefig(fig_name)
    plt.close()

    print("Plotted %s" % fig_name)


def plot_money_time_series(args):
    """
    Plot results of experiments run on money time series.

    :param args:    Argument of scripts
    """
    # Get data from all experiments
    data = u.load_from_folder(args.folder, args.label, args.metric)

    # Load Sidekick results if we want accuracy
    if args.metric == "accuracy" and args.sidekick:
        sidekick_median = u.load_file("%s/vincent_reduced_dataset_accuracy_median.pkl" % args.folder) # vincent_accuracy_median.pkl
        sidekick_std = u.load_file("%s/vincent_reduced_dataset_accuracy_std.pkl" % args.folder) # vincent_accuracy_std.pkl

        # QUICK FIX
        sidekick_median.append(100)
        sidekick_std.append(0)

        # Construct data object
        sidekick_data = {"x": np.linspace(0.025, 1.0, 40),
                         "y_value": sidekick_median,
                         "y_err": sidekick_std,
                         "plot_label": "sidekick"}
        data.append(sidekick_data)

    # Construct plot description
    title, x_label, y_label = _build_plot_description(args.metric)

    # Plot experiment results
    plot_data(data, title, x_label, y_label, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', default="data/average-increment-granularity", help="Folder to load data from")
    parser.add_argument('--label', default="linear-regression", help="Label to identify the experiment")
    parser.add_argument('--metric', default="accuracy", help="Which metric to use")
    parser.add_argument('--sidekick', default=False, help="Plot sidekick results")

    args = parser.parse_args()

    u.args_sanity_check(vars(args))

    plot_money_time_series(args)
