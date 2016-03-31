import pickle as cp
import glob


class Utils:

    def __init__(self, data_dir=""):
        self.data_dir = data_dir

    def _construct_file_name(self, args):
        """
        Construct a file name from the dict of arguments, with the following conventions:
         - Arguments name are CamelCased
         - Arguments name cannot contain dash "-", underscores "_" or equal sign "="
         - Arguments value is dash-cased
         - Arguments value cannot contain underscores "_" or equal sign "="
         - Arguments are sorted alphabetically
         - "label" is mandatory
         - "metric", if present, will be added after the label

        :param args:    Dict of arguments used in the file name
        :return:        File name as string
        """
        # Set the label
        file_name = self.data_dir + args['label']

        # Add the metric if present
        if "metric" in args:
            file_name += "_" + args["metric"]

        # Add the alphabetically-sorted arguments
        keys = sorted(args.keys())
        for k in keys:
            if k != "label" and k != "metric":
                file_name += "_%s=%s" % (k, args[k])

        return file_name + ".pkl"

    def save_file(self, data, file_name):
        """
        Save data using pickle.

        :param data:        Data to save
        :param file_name:   File name
        """
        with open(file_name, 'wb') as f:
                cp.dump(data, f)

    def load_file(self, file_name):
        """
        Load using pickle.

        :param file_name: File name
        :return:
        """
        with open(file_name, 'rb') as f:
            return cp.load(f)

    def save_args(self, data, args):
        """
        Save data based on some arguments.

        :param data:    Data to be saved to disk
        :param args:    Dict of arguments used to create the name
        """
        file_name = self._construct_file_name(args)

        # Save file to disk
        self.save_file(data, file_name)

    def load_args(self, args):
        """
        Load binary based on some arguments.

        :param args:    Dict of arguments used in the file name
        :return:        The saved object
        """
        file_name = self._construct_file_name(args)

        # Load file from disk
        self.load_file(file_name)

    def load_from_folder(self, folder, label, metric):
        """
        Load all pickle files from folder.

        :param folder:  Source folder
        :return:        Loaded data in list
        """
        data = []
        for file in glob.glob(folder + "/%s_%s*.pkl" % (label, metric)):
            data.append(self.load_file(file))
        return data

    def args_sanity_check(self, args):
        """
        Check that the args name and value do not contain invalid symbol.

        :param args:    Dict of arguments
        :raise:         ArgumentError
        """
        label = args.get("label", None)
        if label is None or label == "":
            raise ArgumentError("Missing label in arguments")
        for k, v in args.items():
            if "-" in k or "_" in k or "=" in k:
                raise ArgumentError("Invalid symbol in argument name ('%s')" % k)
            if v is str:
                if "_" in v or "=" in v:
                    raise ArgumentError("Invalid symbol in argument value ('%s')" % k)


class Error(Exception):
    def __init__(self, message):
        self.message = message


class ArgumentError(Error):
    def __init__(self, message):
        super(ArgumentError, self).__init__(message)