from .sample import Sample
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (18,8)
# matplotlib.rcParams['font.size'] = 16
# matplotlib.rcParams['legend.fontsize'] = 16

class Project(Sample):
    def __init__(self, project, status):
        super(Sample, self).__init__()
        self.data_dir += "/sidekick"
        self.project_id = str(project[0])
        self.goal = project[1]
        self.successful = project[2] == 1
        self.start_date = project[3]
        self.deadline = project[4]
        self.time = np.array([s[0] for s in status])
        self.money = np.array([s[1] for s in status])
        self.backers = np.array([s[2] for s in status])

    def __str__(self):
        return "Project %s is %s" % (self.project_id, "successful" if self.successful else "failed")

    def split(self, x, y, threshold=0.8):
        """
        Keep k percent of the outputs y for training and 100-k percent for testing.

        :param x:           Input values
        :param y:           Series to split
        :param threshold:   Threshold to keep as as training
        :return:            x_train, y_train, x_test, y_test
        """
        x_train, x_test = self._split(x, threshold)
        y_train, y_test = self._split(y, threshold)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def subsample(y):
        """
        Keep only values where jumps occur.

        :param y:   Outputs series
        :return:    x, y
        """
        clean = [(0, 0)]
        seen = [0]
        for i in range(1, len(y)):
            if y[i] != y[i - 1]:
                if i - 1 not in seen:
                    clean.append((i - 1, y[i - 1]))
                    seen.append(i - 1)
                clean.append((i, y[i]))
                seen.append(i)
        if i not in seen:
            clean.append((i, y[i - 1]))
        l = zip(*clean)
        X = l[0]
        Y = l[1]
        return X, Y

    @staticmethod
    def difference_series(Y):
        """
        Keep only the jumps.

        :param Y:   Time series
        :return:    Indices, Y
        """
        diff = [(0, 0)]
        for i in range(1, len(Y)):
            if Y[i] != Y[i - 1]:
                diff.append((i, Y[i] - Y[i - 1]))
        l = zip(*diff)
        X = l[0]
        Y = l[1]
        return X, Y

    @staticmethod
    def resample(Y, N=300):
        """
        Keep N random samples.

        :param Y:   Time series
        :param N:   Number of samples to keep
        :return:    X, Y
        """
        X = np.arange(1000)
        X_new = np.random.choice(X, N)
        X_new.sort()
        return X_new, Y[X_new]

    def plot(self, file_name=None):
        """
        Plot status of current project.

        :return:
        """
        x = np.linspace(0, 1, len(self.money))

        print("Goal: $%s" % self.goal)
        print("Pledged: $%s" % int(self.money[-1] * self.goal))

        # Pledged money
        fig, ax1 = plt.subplots()
        ax1.plot(x, self.money * self.goal, 'c-')
        ax1.set_xlabel('Relative time')
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel('Pledged money ($)', color='c')
        ax1.set_ylim([0, max(ax1.get_ylim()[1] + 1, self.goal + self.goal * 0.05)])
        for tl in ax1.get_yticklabels():
            tl.set_color('c')

        # Number of backers
        ax2 = ax1.twinx()
        ax2.plot(x, self.backers, 'm-')
        ax2.set_ylabel('Number of backers', color='m')
        ax2.set_ylim([0, ax2.get_ylim()[1] + 1])
        for tl in ax2.get_yticklabels():
            tl.set_color('m')

        # Goal
        line = ax1.axhline(y=self.goal, color="k", linestyle="--", linewidth=2, zorder=0)
        ax1.text(0.01, self.goal + self.goal * 0.01, "Goal")

        plt.title("Trajectory of project %s" % self.project_id)
        if file_name:
            plt.savefig(file_name)
        plt.show()

    def save(self):
        self._save_binary("%s/project_%s.pkl" % (self.data_dir, self.project_id), self)
