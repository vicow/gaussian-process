from __future__ import print_function
from .dataset import Dataset, DatasetError
from .project import Project
import numpy as np


class Sidekick(Dataset):
    def __init__(self, data_dir=None, seed=None):
        super(Sidekick, self).__init__(self.__class__.__name__)
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir += "/sidekick"
        if seed:
            np.random.seed(seed)

    ###############
    # Access sample
    ###############

    def __getitem__(self, project_id):
        """
        Get a project in the Sidekick dataset.

        Allows:
        >>>> sk = Sidekick()
        >>>> project = sk['14035777']

        :param project_id:
        :return:
        """
        try:
            return self._load_project(project_id)
        except ProjectNotFound:
            if not self.data:
                self.load()
            for project in self.data:
                if project_id == project.project_id:
                    return project
            raise ProjectNotFound("Project %s not found")

    def __iter__(self):
        """
        Iterator over projects in Sidekick dataset.

        Allows:
        >>>> sk = Sidekick()
        >>>> for project in sk:
        >>>>    print project

        :return:
        """
        for project in self.data:
            yield project

    ###########
    # Load data
    ###########

    def load(self, light=False):
        """
        Load Sidekick data.
        """
        if light:
            print("Loading light data set (1000 data points)...")
            self.data = self._load_binary('%s/light.pkl' % self.data_dir)
        else:
            print("Loading data set...")
            self.data = self._load_binary('%s/complete.pkl' % (self.data_dir, ))

            # print('Loading projects...')
            # projects = np.load('%s/projects.npy' % (self.data_dir, ))
            # print('Loading statuses...')
            # statuses = self._load_binary('%s/statuses.pkl' % (self.data_dir, ))
            # assert(len(projects) == len(statuses))
            #
            # print('Converting to project instances...')
            # for i, p in enumerate(projects):
            #     project = Project(p, statuses[i])
            #     self.data.append(project)
            #
            # self._save_binary("%s/complete.pkl" % self.data_dir, self.data)

            # Convert to numpy arrays if needed
            # self.statuses = np.array(self.statuses)

        print("Data loaded.")

    def _load_project(self, project_id):
        """
        Load a saved project.

        :param project_id:  Id of project to load
        :return:
        """
        try:
            return self._load_binary("%s/project_%s.pkl" % (self.data_dir, project_id))
        except IOError:
            raise ProjectNotFound("Project %s not found" % project_id)


    def choose_n_projects(self, n=100, seed=0):
        """
        Choose n projects randomly form the whole list of projects.

        :param n:   Number of projects to extract. If None or negative, take the whole list.
        :return:    Corresponding random indices, list of n projects
        """
        np.random.seed(seed)
        ind = np.arange(len(self.data))
        np.random.shuffle(ind)
        ind = ind[:n]
        data = np.array(self.data)  # To support n-ary indices
        return data[ind]

    def successful(self):
        """
        Keep only the successful projects.

        *Note:* Currently, 142 projects in the data have a status of success but their final amount of pledged money
        is less than 1.0. This is due to bug in the crawler that was down for a few days between the 24th and the 27th
        of December 2012. The success state is correct, but the amount of money is wrong. We ignore these projects.

        :return: List of successful projects
        """
        return [p for p in self if p.successful and p.money[-1] >= 1.0]

    def failed(self):
        return [p for p in self if not p.successful]

    def histogram(self):
        # X, Y = project.difference_series(money)
        # hist, bins = np.histogram(Y, bins=50)
        # width = 0.7 * (bins[1] - bins[0])
        # center = (bins[:-1] + bins[1:]) / 2
        # plt.bar(center, hist, align='center', width=width)
        # plt.yscale('log')
        # plt.show()
        pass


class ProjectNotFound(DatasetError):
    def __init__(self, message):
        self.message = message