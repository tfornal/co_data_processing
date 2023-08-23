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


pd.options.mode.chained_assignment = None  # default='warn'


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
        self.fpm_object = FilePathManager(self.element, None)
        self.fobject = FileInformationCollector(self.path)

        self.file_list = self._get_file_list()
        self.file_sizes = self._get_file_sizes()
        self.date = self._get_date_from_files()
        self.triggers_df = self._get_triggers()
        self.files_info = self.make_df()
        self.utc_time = self.get_UTC_time()

        self.files_into = self.assign_discharge_nr()
        ## wadliwa funkcja ponizej? - nie mozna wykonac - nie mozna stworzyc obiektu;
        self.get_frequency()
        if savefile:
            print("tak")
            self.save_file()

    def _get_file_list(self):
        return self.fobject.file_list

    def _get_file_sizes(self):
        return self.fobject.file_sizes

    def _get_date_from_files(self):
        return self.fobject.date

    def _get_triggers(self):
        t = Triggers(self.date)
        return t.triggers_df

    def retrieve_file_info(self):
        splitted_fnames = []
        for file_name, file_size in zip(self.file_list, self.file_sizes):
            fname_parts = file_name.split("_")
            ## change time format from 12 to 24 hour
            if fname_parts[1].startswith("PM"):
                hour = 12 + int(fname_parts[1][2:][:2])

                fname_parts[1] = f"{hour}{fname_parts[1][4:]}"
            elif fname_parts[1].startswith("AM"):
                fname_parts[1] = fname_parts[1][2:]
            splitted_fnames.append(fname_parts)
        return splitted_fnames

    def make_df(self):
        splitted_fnames = self.retrieve_file_info()
        df = pd.DataFrame(splitted_fnames)
        if len(df.columns) == 3:
            df["type_of_data"] = "spectrum"
        df = df.fillna("spectrum")
        df["file_size"] = self.file_sizes
        df.columns = ["date", "time", "miliseconds", "type_of_data", "file_size"]
        df = df.astype({"file_size": int})
        df.insert(loc=0, column="file_name", value=self.file_list)
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

        date = "20" + date

        # convert European/Berlin timezonee to UTC
        from_zone = tz.gettz("Europe/Berlin")
        to_zone = tz.gettz("UTC")
        discharge_time = (
            datetime.strptime(f"{date} {time} {miliseconds}", "%Y%m%d %H%M%S %f")
            .replace(tzinfo=from_zone)
            .astimezone(to_zone)
        )

        utc_time_in_ns = (
            int(round(calendar.timegm(discharge_time.utctimetuple()))) * 1_000_000_000
            + int(miliseconds) * 1_000_000
        )
        return utc_time_in_ns

    def get_UTC_time(self):
        """
        Calculates the UTC timestamp for each row of `files_info` and adds it as a new column named 'utc_time'.

        Returns:
        -------
        List : int
            The list of UTC timestamps for each row of `files_info`.
        """
        self.files_info["utc_time"] = self.files_info.apply(
            lambda row: self._convert_human_to_UTC_ns(
                row["date"], row["time"], row["miliseconds"]
            ),
            axis=1,
        )
        self.files_info["discharge_nr"] = "-"
        return self.files_info["utc_time"].tolist()

    def assign_discharge_nr(self):
        """
        Assigns the discharge number to each data point in `self.files_info` based on its `utc_time`.

        The discharge number is taken from `self.triggers_df`. For each row in `self.files_info`, this function
        iterates through each row in `self.triggers_df` to find a corresponding discharge number, by checking if the
        `utc_time` of the current row in `self.files_info` falls between the `T0` of the current and previous row
        in `self.triggers_df`.

        If a match is found, the discharge number of the corresponding row in `self.triggers_df` is appended to a list.
        The list is then transformed into a numpy array and used to update the `discharge_nr` column in `self.files_info`.

        If there is no discharge registered during the day, a message is printed with the date.
        """

        ### dodac warunki
        #### SLOW - correct
        dic = {}
        for idx_df_total, row_total in self.files_info.iterrows():
            for (
                idx_df_triggers,
                row_triggers,
            ) in self.triggers_df.iterrows():
                try:
                    if idx_df_triggers == 0:
                        continue
                    if "BGR" in row_total["file_name"]:
                        if (
                            self.triggers_df["T6"].loc[idx_df_triggers - 1]
                            < row_total["utc_time"]
                            < self.triggers_df["T6"].loc[idx_df_triggers]
                        ):
                            dic[idx_df_total] = row_triggers["discharge_nr"]
                            continue

                    elif (row_total.file_size > 10) and (
                        self.triggers_df["T1"].loc[idx_df_triggers - 1]
                        < row_total["utc_time"]
                        < self.triggers_df["T1"].loc[idx_df_triggers]
                    ):
                        dic[idx_df_total] = row_triggers["discharge_nr"] - 1
                        continue
                    elif (row_total.file_size > 10) and (
                        self.triggers_df["T6"].loc[idx_df_triggers - 1]
                        < row_total["utc_time"]
                        < self.triggers_df["T6"].loc[idx_df_triggers]
                    ):
                        dic[idx_df_total] = row_triggers["discharge_nr"]

                    elif (
                        self.triggers_df["T6"].loc[idx_df_triggers - 1]
                        < row_total["utc_time"]
                        < self.triggers_df["T1"].loc[idx_df_triggers]
                    ):
                        dic[idx_df_total] = row_triggers["discharge_nr"]
                except KeyError:
                    dic[idx_df_total] = row_triggers["discharge_nr"]

        files_info_assigned_discharges = self.files_info
        try:
            files_info_assigned_discharges["discharge_nr"].loc[
                np.array([i for i in dic.keys()])
            ] = np.array([i for i in dic.values()])
        except ValueError:
            print(f"\n{self.date} - no discharges registered during the day!\n")
        files_info_assigned_discharges.astype(
            {"discharge_nr": "int32"}, errors="ignore"
        )
        return files_info_assigned_discharges

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
                new_date_string = date_string[
                    2:
                ]  # Zmiana formatu z "YYYYMMDD" na "YYMMDD"
                new_date_integer = str(
                    new_date_string
                )  # Konwersja z powrotem na integer
                return new_date_integer
            else:
                return date_integer

        # Zastosowanie funkcji do kolumny
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

        # fill NaN if no values mathes
        merged_df["frequency"].fillna(200, inplace=True)
        # sort all columns by file_name column
        self.files_info = merged_df.sort_values(by="file_name")

        self.files_info["frequency"] = self.files_info["frequency"].astype(int)

    def save_file(self):
        path = self.fpm_object.discharge_nrs()
        path.mkdir(parents=True, exist_ok=True)
        self.files_info.to_csv(path / f"{self.element}-{self.date}.csv", sep="\t")
        print("Experimental details saved!")


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
    elements = ["C"]
    for element in elements:
        list_of_directories = get_exp_data_subdirs(element)

        for directory in list_of_directories:
            try:
                exp_ass = ExpAssignment(element, directory, savefile=True)
            except ValueError:
                print("Cannot run class (or sth else) - continuing.")
                continue
