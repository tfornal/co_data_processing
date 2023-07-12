import pathlib


class Files:
    """Retrieves information about directories and files inside them to be processed."""

    def __init__(self, path):
        self.path = path
        self.date = self.get_date_from_fnames()
        self.file_list = self.get_file_list()
        self.file_sizes = self.get_file_sizes()

    def get_date_from_fnames(self):
        return self.path.stem

    def get_file_list(self):
        """
        Returns a list of file names in the directory.
        """
        directory = self.path.glob("**/*")
        file_list = [x.stem for x in directory if x.is_file()]

        return file_list

    def get_file_sizes(self):
        """
        Returns a sizes of all files in a given directory.
        """
        directory = self.path.glob("**/*")
        file_sizes = [
            pathlib.Path(x).stat().st_size / 1024 for x in directory if x.is_file()
        ]

        return file_sizes
