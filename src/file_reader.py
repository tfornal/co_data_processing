from __future__ import annotations
import pathlib
import pandas as pd
import natsort


class FilePathManager:
    """Retrieves paths to different folders containing input/output files relative
    to source code."""

    main_path = pathlib.Path(__file__).parent.parent.resolve() / "data"

    @classmethod
    def _get_path(cls, *parts):
        return cls.main_path.joinpath(*parts)

    def get_directory_for_exp_numbers(self, element):
        return self._get_path("discharge_numbers", element)

    def get_directory_for_exp_data(self, element, date):
        return self._get_path("exp_data", element, date)

    def get_directory_for_exp_data_parameters(self):
        return self._get_path("exp_data_parameters")

    def get_directory_for_program_triggers(self):
        return self._get_path("program_triggers")

    def get_directory_for_time_evolutions(self, element, date):
        return self._get_path("time_evolutions", element, date)

    def get_directory_for_images(self, element, date):
        return self._get_path("time_evolutions", element, date, "img")


class FileListExtractor(FilePathManager):
    """Grabs all file names containing experimental and background data
    related to the given experimental discharge number."""

    @classmethod
    def grab_all_file_list(cls, path):
        return list(path.glob("**/*"))

    def select_file_names(self, element, date, exp_nr):
        """Returns list of files that mathes the given date and discharge number."""
        df = pd.read_csv(
            self.get_directory_for_exp_numbers(element) / f"{element}-{date}.csv",
            sep="\t",
        )
        if exp_nr > 0:
            df["discharge_nr"] = df["discharge_nr"].replace("-", "0").astype(int)
            selected_file_names = df.loc[df["discharge_nr"] == exp_nr][
                "file_name"
            ].to_list()
            return selected_file_names


class BackgroundFilesSelector(FileListExtractor):
    """Grabs all file names containing background data
    related to the given experimental discharge number."""

    def _get_all_file_list(self, element, date):
        return list(self.get_directory_for_exp_data(element, date).glob("**/*"))

    def get_bgr_file_list(self, element, date, exp_nr):
        bgr_files = [
            x
            for x in self._get_all_file_list(element, date)
            if "BGR" in x.stem
            and x.stem in self.select_file_names(element, date, exp_nr)
        ]
        return bgr_files


class ExperimentalFilesSelector(FileListExtractor):
    """Grabs all file names containing experimental data
    related to the given experimental discharge number."""

    def grab_discharge_files(self, element, date, exp_nr):
        path = self.get_directory_for_exp_data(element, date)
        file_list = self.grab_all_file_list(path)
        discharge_files = [
            file
            for file in file_list
            if file.stat().st_size > 8000
            and file.stem in self.select_file_names(element, date, exp_nr)
            and "BGR" not in file.stem
        ]
        return discharge_files


class ExperimentalDataExtractor(FileListExtractor):
    def get_discharge_parameters(self, element, date, file_name):
        with open(
            self.get_directory_for_exp_numbers(element) / f"{element}-{date}.csv",
            "r",
        ) as data:
            df = pd.read_csv(
                data,
                sep="\t",
            )
            df = df.astype({"date": int})
            discharge_data = df.loc[df["file_name"] == file_name.stem]
        return discharge_data


class FileInformationCollector:
    """Retrieves list of all files in a directory and their sizes."""

    def _grab_directory_content(cls, path):
        return list(path.glob("**/*"))

    def get_files_names(self, path) -> list:
        """
        Returns a list of file names in the directory.
        """
        files_list = self._grab_directory_content(path)
        files_names = [x.stem for x in files_list if x.is_file()]

        return files_names

    def get_files_sizes(self, path):
        """
        Returns a sizes of all files in a given directory in kilobytes.
        """
        directory = self._grab_directory_content(path)
        files_sizes = [
            pathlib.Path(x).stat().st_size / 1024 for x in directory if x.is_file()
        ]

        return files_sizes
