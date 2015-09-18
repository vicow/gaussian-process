from src.dataset.sidekick import Sidekick


def evaluation():
    sk = Sidekick()
    sk.load()
    projects = sk.extract_n_projects()



if __name__ == '__main__':
    evaluation()