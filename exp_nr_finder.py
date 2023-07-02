import pathlib


class Files:
    """Retrieves information about directories and files inside them to be processed."""

    def __init__(self, path):
        self.directory = path
        self.date = self.directory.stem
        self.file_list, self.file_sizes = self.get_file_list()

    def get_file_list(self):
        """
        Returns a list of file names in the directory.
        """
        directory = self.directory.glob("**/*")
        file_list = [x.stem for x in directory if x.is_file()]
        directory = self.directory.glob("**/*")

        file_sizes = [
            pathlib.Path(x).stat().st_size / 1024 for x in directory if x.is_file()
        ]
        return file_list, file_sizes
