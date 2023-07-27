import pathlib
import pandas as pd


class FilePathManager:
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
        path = self.stem_path / "discharge_numbers" / self.element
        return path

    def experimental_data(self):
        path = self.stem_path / "exp_data" / self.element / self.date
        return path

    def experimental_data_parameters(self):
        path = self.stem_path / "exp_data_parameters"
        return path

    def program_triggers(self):
        path = (
            pathlib.Path(__file__).parent.parent.resolve() / "data" / "program_triggers"
        )
        return path

    def time_evolutions(self):
        path = self.stem_path / "time_evolutions" / self.element / self.date
        return path

    def images(self):
        path = self.stem_path / self.time_evolutions() / "img"
        return path


class FileInformationCollector:
    """Retrieves list of all files in a directory and their sizes."""

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


class FileListExtractor:
    """Grabs all file names containing experimental and background data
    related to the given experimental discharge number."""

    def __init__(self, element, date, discharge_nr):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr

        self.fp = FilePathManager(self.element, self.date)
        self.discharge_nr_file_path = self.fp.discharge_nrs()
        self.exp_data_file_path = self.fp.experimental_data()

        self.all_file_list = self.grab_all_file_list()
        self.selected_file_names = self.select_file_names()

    def grab_all_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def select_file_names(self):
        """Returns list of files that mathes the given date and discharge number."""

        df = pd.read_csv(self.discharge_nr_file_path, sep="\t")
        if self.discharge_nr != 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()
            return selected_file_names


class BackgroundFilesSelector(FileListExtractor):
    """Grabs all file names containing background data
    related to the given experimental discharge number."""

    def __init__(self, element, date, discharge_nr):
        super().__init__(element, date, discharge_nr)
        self.bgr_files = self.grab_bgr_files(discharge_nr)

    def select_file_names(self, discharge_nr):
        """Returns list of files that mathes the given date and discharge number."""

        df = pd.read_csv(self.discharge_nr_file_path, sep="\t")
        if self.discharge_nr != 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()
            return selected_file_names

    def grab_all_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def grab_bgr_files(self):
        bgr_files = [
            x
            for x in self.all_file_list
            if "BGR" in x.stem and x.stem in self.selected_file_names
        ]
        return bgr_files


class DischargeFilesSelector(FileListExtractor):
    """Grabs all file names containing experimental data
    related to the given experimental discharge number."""

    def __init__(self, element, date, discharge_nr):
        super().__init__(element, date, discharge_nr)
        self.discharge_files = self._grab_discharge_files()

    def _grab_discharge_files(self):
        discharge_files = [
            x
            for x in self.all_file_list
            if x.stat().st_size > 8000
            and x.stem in self.selected_file_names
            and "BGR" not in x.stem
        ]
        return discharge_files


class DischargeDataExtractor:
    def __init__(self, element, date, file_name):
        self.element = element
        self.date = date
        self.file_name = file_name
        self.discharge_nr_file_path = self._get_specific_file_path()
        self.discharge_data = self.get_discharge_parameters()

    def _get_specific_file_path(self):
        return FilePathManager(self.element, self.date).discharge_nrs()

    def get_discharge_parameters(self):
        with open(self.discharge_nr_file_path, "r") as data:
            df = pd.read_csv(
                data,
                sep="\t",
                usecols=[
                    "date",
                    "discharge_nr",
                    "file_name",
                    "time",
                    "type_of_data",
                    "file_size",
                    "utc_time",
                    "frequency",
                ],
            )
            df = df.astype({"date": int})
            discharge_data = df.loc[df["file_name"] == self.file_name.stem]
        return discharge_data
