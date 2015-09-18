from dataset import Dataset
import numpy as np


class Sidekick(Dataset):
    def __init__(self):
        super(Sidekick, self).__init__(self.__class__.__name__)
        self.data_dir += "/sidekick"
        self.projects = None
        self.statuses = None

    def load(self):
        """
        Load Sidekick data.
        """
        print 'Loading projects...'
        self.projects = np.load('%s/projects.npy' % (self.data_dir, ))
        print 'Loading statuses...'
        self.statuses = self._load_binary('%s/statuses.pkl' % (self.data_dir, ))

        # Convert to numpy arrays if needed
        # self.statuses = np.array(self.statuses)

        print "Data loaded."

    def extract_project(self, index, save=False):
        """
        Extract a project at the given index.

        :param index:   Index of project
        :param save:    Whether to save the project to disk as <data_dir>/project_<id>.pkl
        :return:
        """

        project = self.projects[index]
        status = self.statuses[index]
        data = {
            "project": project,
            "status": status
        }

        if save:
            project_id = str(project[0])
            print "Saving project %s" % project_id
            self._save_binary("%s/project_%s.pkl" % (self.data_dir, project_id), data)

        return data

    def choose_n_projects(self, n=100):
        """
        Choose n projects randomly form the whole list of projects.

        :param n:   Number of projects to extract. If None or negative, take the whole list.
        :return:    Corresponding random indices, list of n projects
        """
        ind = np.random.randint(len(self.projects), size=(n,))
        return [self.extract_project(i) for i in ind]