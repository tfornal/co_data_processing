from __future__ import annotations
import pathlib
import pandas as pd
import natsort


class FilePathManager:
    """Retrieves paths to different folders containing input/output files relative
    to source code."""

    main_path = pathlib.Path(__file__).parent.parent.resolve() / "data"

    def __init__(self, element: str, date: str):
        self._element = element
        self._date = date

    @property
    def element(self) -> str:
        return self._element

    @property
    def date(self) -> str:
        return self._date

    @classmethod
    def _get_path(cls, *parts):
        return cls.main_path.joinpath(*parts)

    def get_directory_for_exp_numbers(self):
        return self._get_path("discharge_numbers", self.element)

    def get_directory_for_exp_data(self):
        return self._get_path("exp_data", self.element, self.date)

    def get_directory_for_exp_data_parameters(self):
        return self._get_path("exp_data_parameters")

    def get_directory_for_program_triggers(self):
        return self._get_path("program_triggers")

    def get_directory_for_time_evolutions(self):
        return self._get_path("time_evolutions", self.element, self.date)

    def get_directory_for_images(self):
        return self._get_path("time_evolutions", self.element, self.date, "img")


class FileListExtractor(FilePathManager):
    """Grabs all file names containing experimental and background data
    related to the given experimental discharge number."""

    def __init__(self, element, date, discharge_nr):
        self._element = element
        self._date = date
        self._discharge_nr = discharge_nr

    @property
    def element(self) -> str:
        return self._element

    @property
    def date(self) -> str:
        return self._date

    @property
    def discharge_nr(self) -> str:
        return self._discharge_nr

    @classmethod
    def grab_all_file_list(cls, path):
        return list(path.glob("**/*"))

    def select_file_names(self):
        """Returns list of files that mathes the given date and discharge number."""
        df = pd.read_csv(
            self.get_directory_for_exp_numbers() / f"{self.element}-{self.date}.csv",
            sep="\t",
        )
        if self.discharge_nr > 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == self.discharge_nr][
                "file_name"
            ].to_list()
            return selected_file_names


class BackgroundFilesSelector(FileListExtractor):
    """Grabs all file names containing background data
    related to the given experimental discharge number."""

    def _get_all_file_list(self):
        return list(self.get_directory_for_exp_data().glob("**/*"))

    def get_bgr_file_list(self):
        bgr_files = [
            x
            for x in self._get_all_file_list()
            if "BGR" in x.stem and x.stem in self.select_file_names()
        ]
        return bgr_files


class ExperimentalFilesSelector(FileListExtractor):
    """Grabs all file names containing experimental data
    related to the given experimental discharge number."""

    def grab_discharge_files(self):
        path = self.get_directory_for_exp_data()
        file_list = self.grab_all_file_list(path)
        discharge_files = [
            file
            for file in file_list
            if file.stat().st_size > 8000
            and file.stem in self.select_file_names()
            and "BGR" not in file.stem
        ]
        return discharge_files


class FileInformationCollector:
    """Retrieves list of all files in a directory and their sizes."""

    def grab_directory_content(cls, path):
        return list(path.glob("**/*"))

    def get_files_names(self, path) -> list:
        """
        Returns a list of file names in the directory.
        """
        files_list = self.grab_directory_content(path)
        files_names = [x.stem for x in files_list if x.is_file()]

        return files_names

    def get_files_sizes(self, path):
        """
        Returns a sizes of all files in a given directory in kilobytes.
        """
        directory = self.grab_directory_content(path)
        files_sizes = [
            pathlib.Path(x).stat().st_size / 1024 for x in directory if x.is_file()
        ]

        return files_sizes


class ExperimentalDataExtractor:
    def __init__(self, element, date, file_name):
        self.element = element
        self.date = date
        self.file_name = file_name
        self.discharge_nr_file_path = self.get_specific_file_path()
        self.discharge_data = self.get_discharge_parameters()

    def get_specific_file_path(self):
        return FilePathManager(self.element, self.date).get_directory_for_exp_numbers()

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
