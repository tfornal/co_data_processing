## napisac proceduralnie
"""
The code must contain only the data folder containing the corresponding folders 
in "YYMMDD" format. Subsequent sub-folders (with their names in YYMMDD format) 
must contain the data recorded by the C/O monitor system (in *.dat format) .
"""
import calendar
from pathlib import Path
from dateutil import tz
from datetime import datetime

import natsort
import numpy as np
import pandas as pd

from file_reader import FileInformationCollector, FilePathManager
from triggers import Triggers


pd.options.mode.chained_assignment = None


class ParameterReader:
    pass


class ExpAssignment:
    """This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    Note:
    The class makes use of the Triggers class, pd.DataFrame, and the sys and requests libraries.
    """

    def __init__(self, element, path, savefile=False):
        """
        Initializes the ExpAssignment object with the directory path and saves the file if specified.
        """
        self.element = element
        self.date = path.stem

        self.fpm_object = FilePathManager(element, None)
        self.fobject = FileInformationCollector(path)
        self.fname_list = self._get_file_list()
        self.file_sizes = self._get_file_sizes()
        self.triggers_df = self._get_triggers(self.date)
        self.make_df_with_files_info(self.element, savefile)

    def make_df_with_files_info(self, element, savefile):
        self.files_info = self.make_df()
        self._get_utc_time(self.files_info)
        ## why assign another variable?
        self.files_info = self._get_frequency(self.element, self.files_info)
        self._update_acquisition_time(self.files_info)
        self._calc_utc_start_time_ns(self.files_info)
        self._assign_discharge_nrs(self.files_info, self.triggers_df, self.date)
        self._shift_to_T1(self.files_info, self.triggers_df)
        if savefile:
            self.save_file()

    def _get_file_list(self):
        return self.fobject.file_list

    def _get_file_sizes(self):
        return self.fobject.file_sizes

    def _get_triggers(self, date):
        t = Triggers(date)
        return t.triggers_df

    def convert_fname_to_time(self, file_name):
        fname_parts = file_name.split("_")
        time_part = fname_parts[1]

        if time_part.startswith("PM"):
            hour = int(time_part[2:4])
            if hour != 12:
                hour += 12
            fname_parts[1] = f"{hour:02}{time_part[4:]}"
        elif time_part.startswith("AM"):
            if time_part.startswith("12"):
                fname_parts[1] = f"00{time_part[4:]}"
            else:
                fname_parts[1] = time_part[2:]

        return fname_parts

    def get_info_from_fname(self):
        return list(
            map(
                lambda file_name: self.convert_fname_to_time(file_name), self.fname_list
            )
        )

    def _check_if_longer(self, list_of_sublists: list) -> bool:
        any_longer_than_3 = any(len(sublist) > 3 for sublist in list_of_sublists)

        if any_longer_than_3:
            return True

    def make_df(self) -> pd.DataFrame:
        splitted_fnames = self.get_info_from_fname()
        columns = ["date", "time", "miliseconds"]
        if self._check_if_longer(splitted_fnames):
            columns.append("type_of_data")
            df = pd.DataFrame(splitted_fnames, columns=columns)
        else:
            df = pd.DataFrame(splitted_fnames, columns=columns)
            df["type_of_data"] = "spectrum"

        df["file_name"] = self.fname_list
        df["file_size"] = self.file_sizes
        df["date"] = "20" + df["date"]
        df["file_size"] = df["file_size"].astype(int)

        df["type_of_data"].fillna("spectrum", inplace=True)
        new_order = [
            "date",
            "file_name",
            "time",
            "miliseconds",
            "type_of_data",
            "file_size",
        ]
        files_info_df = df[new_order]

        return files_info_df

    def _convert_human_to_utc_ns(self, date: str, time: str, miliseconds: str) -> int:
        """
        Converts a given date and time to UTC timestamp in nanoseconds.

        Parameters:
        ----------
        date : str
            A string representation of the date in YYMMDD format.
        time : str
            A string representation of the time in HHMMSS format.

        Returns:
        -------
        int
            The UTC timestamp in nanoseconds corresponding to the given date and time.
        """
        from_zone = tz.gettz("Europe/Berlin")
        to_zone = tz.gettz("UTC")
        discharge_time = (
            datetime.strptime(f"{date} {time} {miliseconds}", "%Y%m%d %H%M%S %f")
            .replace(tzinfo=from_zone)
            .astimezone(to_zone)
        )

        utc_time_in_ns = int(
            round(calendar.timegm(discharge_time.utctimetuple())) * 1e9
            + int(miliseconds) * 1e6
        )

        return utc_time_in_ns

    def _get_utc_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the UTC timestamp for each row of `files_info` and adds it as a new column named 'utc_time'.

        Returns:
        -------
        List : int
            The list of UTC timestamps for each row of `files_info`.
        """
        df["utc_time"] = df.apply(
            lambda row: self._convert_human_to_utc_ns(
                row["date"], row["time"], row["miliseconds"]
            ),
            axis=1,
        )
        df["utc_time"] = df["utc_time"].astype("int64")
        df["discharge_nr"] = "-"

        return df

    def get_det_size(self, binary_file):
        # nr of rows means the number of collected frames
        binary_file.seek(4)
        bites = binary_file.read(4)
        nrows = int.from_bytes(bites, "little")

        return nrows

    def get_pulse_length(self, element, date):
        path_manager = FilePathManager(element, date)
        path = path_manager.experimental_data()

        file_paths = list(path.glob("**/*"))
        file_list = [x.stem for x in file_paths if x.is_file()]

        dlugosci = []
        for file_path in file_paths:
            with open(file_path, "rb") as binary_file:
                rows_number = self.get_det_size(binary_file)
                dlugosci.append(rows_number)

        return file_list, dlugosci

    def _get_frequency(self, element, files_info_df):
        path_exp_data_params = self.fpm_object.experimental_data_parameters()
        setup_notes = path_exp_data_params / f"{element}-camera_setups.csv"
        with open(setup_notes, "r") as data:
            camera_setups = pd.read_csv(
                data, sep=",", usecols=["date", "discharge_nr", "ITTE_frequency"]
            )
            camera_setups = camera_setups.astype({"date": int})

        # both dataframes indexed by "date" and "discharge_nr" columns
        camera_setups = camera_setups.set_index(["date", "discharge_nr"])
        files_info_df = files_info_df.set_index(["date", "discharge_nr"])

        # merge both dataframes by indexed columns
        merged_df = files_info_df.merge(
            camera_setups, left_index=True, right_index=True, how="left"
        )
        merged_df["frequency"] = merged_df["ITTE_frequency"]
        merged_df.drop("ITTE_frequency", axis=1, inplace=True)
        merged_df["frequency"].fillna(200, inplace=True)

        files_info_df = merged_df.sort_values(by="file_name")
        files_info_df["frequency"] = files_info_df["frequency"].astype(int)
        file_list, pulse_lengths = self.get_pulse_length(self.element, self.date)
        data = {"file_name": file_list, "frames_amount": pulse_lengths}
        df2 = pd.DataFrame(data)
        files_info_df = files_info_df.reset_index().merge(df2, on="file_name")

        return files_info_df

    def _update_acquisition_time(self, files_info_df):
        dt = 1 / files_info_df["frequency"]
        time = files_info_df["frames_amount"] * dt
        files_info_df["acquisition_time"] = time.round(2)

    def _calc_utc_start_time_ns(self, files_info_df):
        files_info_df["utc_start_time"] = files_info_df["utc_time"] - (
            1e9 * files_info_df["acquisition_time"]
        ).astype("int64")

    def _assign_discharge_nrs(self, files_info_df, triggers_df, date):
        files_info_df["discharge_nr"] = 0
        dictionary = {}

        for idx_df_total, row_total in files_info_df.iterrows():
            utc_time = row_total["utc_time"]
            discharge_nr = None

            for idx_df_triggers, row_triggers in triggers_df.iterrows():
                t1 = row_triggers["T1"]
                t6 = row_triggers["T6"]
                discharge_nr_trigger = row_triggers["discharge_nr"]
                if (
                    "BGR" not in row_total["file_name"]
                    and idx_df_triggers != 0
                    and row_total["file_size"] > 10
                    and (
                        # checks if the data were saved even 15s after T6
                        # (just in case file saving took too much time)
                        (t1 < utc_time < t6 + 15e9)
                        or (
                            t1 < row_total["utc_start_time"] < utc_time
                            and t1 < row_total["utc_time"] < t6
                        )
                    )
                ):
                    discharge_nr = discharge_nr_trigger
                    break

                elif "BGR" in row_total["file_name"] and (t6 > utc_time):
                    discharge_nr = discharge_nr_trigger
                    break

            if discharge_nr is not None:
                dictionary[idx_df_total] = discharge_nr
            else:
                row_total["type_of_data"] = "trash"
        if dictionary:
            files_info_df["discharge_nr"].loc[
                np.array(list(dictionary.keys()))
            ] = np.array(list(dictionary.values()))
            indexes_to_assign = files_info_df.index[
                ~files_info_df.index.isin(dictionary)
            ]
            files_info_df["type_of_data"].loc[indexes_to_assign] = "trash"
        else:
            print(f"\n{date} - no discharges registered during the day!\n")

        files_info_df["discharge_nr"] = files_info_df["discharge_nr"].astype(
            {"discharge_nr": "int32"}, errors="ignore"
        )

    def _shift_to_T1(self, files_info_df, triggers_df):
        calibrated_start_times = {}
        discharge_nr = 0
        save_time_offset = 0

        for idx_df_total, row_total in files_info_df.iterrows():
            for idx_df_triggers, row_triggers in triggers_df.iterrows():
                if (
                    row_total["discharge_nr"] == row_triggers["discharge_nr"]
                    and "BGR" not in row_total["file_name"]
                ):
                    offset = row_total["utc_start_time"] - row_triggers["T1"]
                    if row_total["discharge_nr"] != discharge_nr:
                        save_time_offset = offset
                        discharge_nr = row_total["discharge_nr"]
                    calibrated_start_times[idx_df_total] = save_time_offset

        files_info_df["new_time"] = [0] * len(files_info_df)
        idx = list(calibrated_start_times.keys())
        offset = list(calibrated_start_times.values())
        files_info_df["new_time"].loc[idx] = (
            files_info_df["utc_start_time"].loc[idx]
            - offset
            + files_info_df["acquisition_time"] * 1e9
        )
        files_info_df["new_time"] = files_info_df["new_time"].astype("int64")

    def save_file(self):
        path = self.fpm_object.discharge_nrs()
        path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_csv(path / f"{self.element}-{self.date}.csv", sep="\t")

        print(f"{self.element}_{self.date} - Experimental details saved!")


def get_exp_data_subdirs(element):
    """
    Retrieves all subdirectories in a given directory.
    """
    path = Path(__file__).parent.parent.resolve() / "data" / "exp_data" / element
    sub_dirs = [f for f in path.iterdir() if f.is_dir() and f.name[0] != (".")]

    return natsort.os_sorted(sub_dirs)


if __name__ == "__main__":
    elements = ["C"]  # , "O"]
    for element in elements:
        list_of_directories = get_exp_data_subdirs(element)
        for directory in list_of_directories:
            # if "20230117" in str(directory):
            try:
                exp_ass = ExpAssignment(element, directory, savefile=True)
            except ValueError:
                print(f" {directory} - Cannot process the data - continuing.")
                continue
