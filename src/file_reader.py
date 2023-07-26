import pathlib
import pandas as pd


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

    def images(self):
        path = self.stem_path / self.time_evolutions() / "img"
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


class ExperimentalFiles:
    def __init__(self, element, date, discharge_nr):
        self.element = element
        self.date = date
        self.discharge_nr = discharge_nr

        self.fp = self._get_file_path_object()
        self.exp_data_file_path = self._get_exp_data_file_path()
        self.file_list = self._grab_file_list()
        self.discharge_nr_file_path = self._get_specific_file_path()
        self.selected_file_names = self._select_file_names()
        self.bgr_files = self._grab_bgr_files()
        self.discharge_files = self._grab_discharge_files()

    def _get_file_path_object(self):
        return FilePaths(self.element, self.date)

    def _get_exp_data_file_path(self):
        return self.fp.experimental_data()

    def _grab_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def _grab_bgr_files(self):
        bgr_files = [
            x
            for x in self._grab_file_list()
            if "BGR" in x.stem in self.selected_file_names
        ]
        return bgr_files

    def _get_specific_file_path(self):
        return self.fp.discharge_nrs()

    def _select_file_names(self):
        df = pd.read_csv(self.discharge_nr_file_path, sep="\t")
        if self.discharge_nr != 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()

            return selected_file_names

    def _grab_discharge_files(self):
        discharge_files = [
            x
            for x in self.file_list
            if x.stat().st_size > 8000
            and x.stem in self.selected_file_names
            and "BGR" not in x.stem
        ]

        return discharge_files
