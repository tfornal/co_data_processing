from __future__ import annotations
import pathlib
import pandas as pd
import natsort


class FilePathManager:
    """Retrieves paths to different folders containing input/output files relative
    to source code."""

    def __init__(self, element=None, date=None):
        self._element = element
        self._date = date

    @property
    def element(self) -> str:
        return self._element

    @property
    def date(self) -> str:
        return self._date

    def _get_path(self, *parts):
        main_path = stem_path = pathlib.Path(__file__).parent.parent.resolve() / "data"
        return main_path.joinpath(*parts)

    def get_exp_numbers_directory(self):
        return self._get_path("discharge_numbers", self.element)

    def get_exp_data_directory(self):
        return self._get_path("exp_data", self.element, self.date)

    def get_exp_data_params_directory(self):
        return self._get_path("exp_data_parameters")

    def get_program_triggers_directory(self):
        return self._get_path("program_triggers")

    def get_time_evolutions_directory(self):
        return self._get_path("time_evolutions", self.element, self.date)

    def get_images_directory(self):
        return self._get_path("time_evolutions", self.element, self.date, "img")


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
        self._element = element
        self._date = date
        self._discharge_nr = discharge_nr

        self.fp = FilePathManager(self.element, self.date)
        self.discharge_nr_file_path = self.fp.get_exp_numbers_directory()
        self.exp_data_file_path = self.fp.get_exp_data_directory()
        self.all_file_list = self.grab_all_file_list()
        self.selected_file_names = self.select_file_names()

    @property
    def element(self) -> str:
        return self._element

    @property
    def date(self) -> str:
        return self._date

    @property
    def discharge_nr(self) -> str:
        return self._discharge_nr

    def grab_all_file_list(self):
        return list(self.exp_data_file_path.glob("**/*"))

    def select_file_names(self):
        """Returns list of files that mathes the given date and discharge number."""
        df = pd.read_csv(
            self.discharge_nr_file_path / f"{self.element}-{self.date}.csv", sep="\t"
        )
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
        self.bgr_files = self.grab_bgr_files()

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
        self.discharge_nr_file_path = self.get_specific_file_path()
        self.discharge_data = self.get_discharge_parameters()

    def get_specific_file_path(self):
        return FilePathManager(self.element, self.date).get_exp_numbers_directory()

    def get_discharge_parameters(self):
        with open(
            self.discharge_nr_file_path / f"{self.element}-{self.date}.csv", "r"
        ) as data:
            df = pd.read_csv(
                data,
                sep="\t",
            )
            df = df.astype({"date": int})
            discharge_data = df.loc[df["file_name"] == self.file_name.stem]
        return discharge_data
