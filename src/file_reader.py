import pathlib


class FilePaths:
    """Retrieves paths to different folders containing input/output files relative
    to source code."""

    def __init__(self, element, date):
        self.element = element
        self.date = date
        self.stem_path = self.main_path()

    def main_path(self):
        stem_path = pathlib.Path(__file__).parent.parent.resolve() / "data"
        return stem_path

    def discharge_nrs(self):
        path = (
            self.stem_path
            / "discharge_numbers"
            / self.element
            / f"{self.element}-{self.date}.csv"
        )
        return path

    def experimental_data(self):
        path = self.stem_path / "exp_data" / self.element / self.date
        return path

    def time_evolutions(self):
        path = self.stem_path / "time_evolutions" / self.element / self.date
        return path


class Files:
    """Retrieves information about directories and files to be processed."""

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
