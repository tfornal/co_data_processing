## napisac proceduralnie
"""
The code must contain only the data folder containing the corresponding folders 
in "YYMMDD" format. Subsequent sub-folders (with their names in YYMMDD format) 
must contain the data recorded by the C/O monitor system (in *.dat format) .
"""
import calendar
import pathlib
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
        self.path = path
        self.date = path.stem

        self.fpm_object = FilePathManager(self.element, None)
        self.fobject = FileInformationCollector(self.path)

        self.fname_list = self._get_file_list()
        self.file_sizes = self._get_file_sizes()
        self.triggers_df = self._get_triggers(self.date)
        self.files_info = self.make_df()
        self.get_UTC_time(self.files_info)
        self.get_frequency()

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
        df = df[new_order]

        return df

    def _convert_human_to_UTC_ns(self, date: str, time: str, miliseconds: str) -> int:
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

    def get_UTC_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the UTC timestamp for each row of `files_info` and adds it as a new column named 'utc_time'.

        Returns:
        -------
        List : int
            The list of UTC timestamps for each row of `files_info`.
        """
        df["utc_time"] = df.apply(
            lambda row: self._convert_human_to_UTC_ns(
                row["date"], row["time"], row["miliseconds"]
            ),
            axis=1,
        )
        df["utc_time"] = df["utc_time"].astype("int64")
        df["discharge_nr"] = "-"

        return df

    def get_frequency(self):
        path = self.fpm_object.experimental_data_parameters()
        setup_notes = path / f"{self.element}-camera_setups.csv"
        with open(setup_notes, "r") as data:
            df = pd.read_csv(
                data, sep=",", usecols=["date", "discharge_nr", "ITTE_frequency"]
            )
            df = df.astype({"date": int})

        def change_dates_format(date_integer):
            date_string = str(date_integer)
            if len(date_string) == 8:
                new_date_string = date_string[2:]
                new_date_integer = str(new_date_string)
                return new_date_integer
            else:
                return date_integer

        df["date"] = df["date"].apply(change_dates_format)

        # both dataframes indexed by "date" and "discharge_nr" columns
        df = df.set_index(["date", "discharge_nr"])
        self.files_info = self.files_info.set_index(["date", "discharge_nr"])

        # merge both dataframes by indexed columns
        merged_df = self.files_info.merge(
            df, left_index=True, right_index=True, how="left"
        )
        merged_df["frequency"] = merged_df["ITTE_frequency"]
        merged_df.drop("ITTE_frequency", axis=1, inplace=True)

        merged_df["frequency"].fillna(200, inplace=True)

        self.files_info = merged_df.sort_values(by="file_name")
        self.files_info["frequency"] = self.files_info["frequency"].astype(int)

        # def get_frames_nr():
        #     ### TODO - dodac kolumne - dlugosc wyladowania
        #     # from intensity import Intensity
        def get_det_size(binary_file):
            binary_file.seek(0)
            bites = binary_file.read(4)
            ncols = int.from_bytes(bites, "little")

            binary_file.seek(4)
            bites = binary_file.read(4)
            nrows = int.from_bytes(bites, "little")
            return nrows, ncols

        def get_pulse_length():
            path = FilePathManager(
                element=self.element, date=self.date
            ).experimental_data()
            directory = path.glob("**/*")
            file_paths = [x for x in directory if x.is_file()]
            directory = path.glob("**/*")
            file_list = [x.stem for x in directory if x.is_file()]
            # assert self.file_list == file_list, "Listy NIE SA SOBIE ROWNE!"
            dlugosci = []
            for i in file_paths:
                with open(i, "rb") as binary_file:
                    # print(i)
                    rows_number, _ = get_det_size(binary_file)
                    dlugosci.append(rows_number)
            return file_list, dlugosci

        file_list, dlugosci = get_pulse_length()  ###
        data = {"file_name": file_list, "frames_amount": dlugosci}
        df2 = pd.DataFrame(data)

        self.files_info = self.files_info.reset_index().merge(df2, on="file_name")
        # breakpoint()

        def calc_acquisition_time():
            dt = 1 / self.files_info["frequency"]
            time = self.files_info["frames_amount"] * dt
            self.files_info["acquisition_time"] = time.round(2)

        def calc_start_utc():
            self.files_info["utc_start_time"] = self.files_info["utc_time"] - (
                1_000_000_000 * self.files_info["acquisition_time"]
            ).astype("int64")

        def check_if_between_triggers():
            self.files_info["discharge_nr"] = 0
            dic = {}
            for idx_df_total, row_total in self.files_info.iterrows():
                for (
                    idx_df_triggers,
                    row_triggers,
                ) in self.triggers_df.iterrows():
                    try:
                        if idx_df_triggers == 0:
                            continue

                        if (row_total.file_size > 10) and (
                            (
                                self.triggers_df["T1"].loc[idx_df_triggers]
                                < (row_total["utc_start_time"] or row_total["utc_time"])
                                < self.triggers_df["T6"].loc[idx_df_triggers]
                            )
                            or (
                                (
                                    row_total["utc_start_time"]
                                    < self.triggers_df["T1"].loc[idx_df_triggers]
                                    < row_total["utc_time"]
                                )
                                and (
                                    row_total["utc_start_time"]
                                    < self.triggers_df["T6"].loc[idx_df_triggers]
                                    < row_total["utc_time"]
                                )
                            )
                        ):
                            dic[idx_df_total] = row_triggers["discharge_nr"]
                            continue
                        elif "BGR" in row_total["file_name"]:
                            if (
                                self.triggers_df["T6"].loc[idx_df_triggers - 1]
                                < row_total["utc_time"]
                                < self.triggers_df["T6"].loc[idx_df_triggers]
                            ):
                                dic[idx_df_total] = row_triggers["discharge_nr"]
                                continue
                        row_total["type_of_data"] == "Trash"
                    except KeyError:
                        dic[idx_df_total] = row_triggers["discharge_nr"]

            try:
                self.files_info["discharge_nr"].loc[
                    np.array([i for i in dic.keys()])
                ] = np.array([i for i in dic.values()])
                indexes_to_assign = self.files_info.index[
                    ~self.files_info.index.isin(dic)
                ]
                # breakpoint()
                self.files_info["type_of_data"].loc[indexes_to_assign] = "trash"
            except ValueError:
                print(f"\n{self.date} - no discharges registered during the day!\n")
            self.files_info.astype({"discharge_nr": "int32"}, errors="ignore")
            # return files_info_assigned_discharges

        def shift_to_T1():
            callibrated_start_times = {}
            numer_wyladowania = 0
            off = 0
            for idx_df_total, row_total in self.files_info.iterrows():
                for (
                    idx_df_triggers,
                    row_triggers,
                ) in self.triggers_df.iterrows():
                    if (
                        row_total["discharge_nr"] == row_triggers["discharge_nr"]
                        and "BGR" not in row_total["file_name"]
                    ):
                        if row_total["discharge_nr"] != numer_wyladowania:
                            offset_zapisu = (
                                row_total["utc_start_time"] - row_triggers["T1"]
                            )
                            off = offset_zapisu
                            numer_wyladowania = row_total["discharge_nr"]
                            callibrated_start_times[idx_df_total] = offset_zapisu
                        else:
                            callibrated_start_times[idx_df_total] = off
            self.files_info["new_time"] = [0] * len(self.files_info)
            idx = list(callibrated_start_times.keys())
            offset = list(callibrated_start_times.values())
            self.files_info["new_time"].loc[idx] = (
                self.files_info["utc_start_time"].loc[idx]
                - offset
                + self.files_info["acquisition_time"] * 1e9
            )
            self.files_info["new_time"] = self.files_info["new_time"].astype("int64")

        calc_acquisition_time()
        calc_start_utc()
        check_if_between_triggers()
        shift_to_T1()

    def save_file(self):
        path = self.fpm_object.discharge_nrs()
        path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_csv(path / f"{self.element}-{self.date}.csv", sep="\t")

        print(f"{self.element}_{self.date} - Experimental details saved!")


def get_exp_data_subdirs(element):
    """
    Retrieves all subdirectories in a given directory.
    """
    path = (
        pathlib.Path(__file__).parent.parent.resolve() / "data" / "exp_data" / element
    )
    sub_dirs = [f for f in path.iterdir() if f.is_dir() and f.name[0] != (".")]

    return natsort.os_sorted(sub_dirs)


if __name__ == "__main__":
    elements = ["C"]  # , "O"]
    for element in elements:
        list_of_directories = get_exp_data_subdirs(element)
        for directory in list_of_directories:
            if "20230117" in str(directory):
                try:
                    exp_ass = ExpAssignment(element, directory, savefile=True)
                except ValueError:
                    print(f" {directory} - Cannot process the data - continuing.")
                    continue
