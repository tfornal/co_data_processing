import pathlib
from dateutil import tz
from datetime import datetime
import numpy as np
import calendar
import pandas as pd
from file_reader import Files
from trigger_reader import Triggers

pd.options.mode.chained_assignment = None  # default='warn'


class ExpAssignment:
    """This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    This class performs the assignment of experiment numbers to files based on the UTC time they were created.

    Note:
    The class makes use of the Triggers class, pd.DataFrame, and the sys and requests libraries.
    """

    def __init__(
        self, element, path, date, file_list, file_sizes, triggers_df, savefile=False
    ):
        self.element = element
        self.date = date
        self.file_list = file_list
        self.file_sizes = file_sizes
        self.triggers_df = triggers_df
        self.all_files_info_df = self._make_df()
        self.utc_time = self._get_utc_time()
        self.assign_exp_nr()

        self.assign_frequency()
        self.cammera_frequency = self.get_frequency(self.date, 61)

        if savefile:
            self.save_file()

    def _make_df(self):
        """
        Returns a DataFrame containing information about all files collected at
        a given time in the considered directory along with their trigger time
        T0 and the assigned discharge number.
        """
        # Split file names and process time information
        splitted_fnames = []
        for file_name, file_size in zip(self.file_list, self.file_sizes):
            parts = file_name.split("_")
            if parts[1].startswith("PM"):
                hour = 12 + int(parts[1][2:][:2])
                parts[1] = f"{hour}{parts[1][4:]}"
            elif parts[1].startswith("AM"):
                parts[1] = parts[1][2:]
            splitted_fnames.append(parts)
        # Create the DataFrame containing assigned discharge numbers
        time_data_from_files = [i for i in splitted_fnames]
        # print(time_data_from_files)

        df = pd.DataFrame(time_data_from_files)
        df.drop(columns=df.columns[-2], axis=1, inplace=True)
        if len(df.columns) < 3:
            df["type_of_data"] = None
        df["file_size"] = self.file_sizes
        df.columns = ["date", "time", "type_of_data", "file_size"]
        df = df.astype({"file_size": int})
        df.insert(loc=0, column="file_name", value=self.file_list)

        return df

    def _get_utc_time(self):
        """
        Calculates the UTC timestamp for each row of `all_files_info_df` and adds it as a new column named 'utc_time'.

        Returns:
        -------
        List : int
            The list of UTC timestamps for each row of `all_files_info_df`.
        """
        self.all_files_info_df["utc_time"] = self.all_files_info_df.apply(
            lambda row: self._convert_to_UTC(row["date"], row["time"]), axis=1
        )

        self.all_files_info_df["discharge_nr"] = "-"

        return self.all_files_info_df["utc_time"].tolist()

    def _convert_to_UTC(self, date: str, time: str) -> int:
        """
        Converts a given date and time to UTC timestamp.

        Parameters:
        ----------
        date : str
            A string representation of the date in YYMMDD format.
        time : str
            A string representation of the time in HHMMSS format.

        Returns:
        -------
        int
            The UTC timestamp corresponding to the given date and time.
        """
        date = "20" + date

        # convert European/Berlin timezonee to UTC
        from_zone = tz.gettz("Europe/Berlin")
        to_zone = tz.gettz("UTC")
        discharge_time = (
            datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
            .replace(tzinfo=from_zone)
            .astimezone(to_zone)
        )

        # convert UTC time to ns
        utc_time_in_ns = (
            int(round(calendar.timegm(discharge_time.utctimetuple()))) * 1_000_000_000
            + discharge_time.microsecond * 1_000
        )

        return utc_time_in_ns

    def find_record(self):
        pass

    def calc_start_time(self):
        pass

    def get_frequency(self, date, exp_nr):
        data_file = (
            pathlib.Path(__file__).parent.parent.resolve()
            / "__Experimental_data"
            / f"{self.element}-camera_setups.csv"
        )
        with open(data_file, "r") as data:
            df = pd.read_csv(
                data, sep=",", usecols=["Date", "Pulse number", "ITTE (Hz)"]
            )
            df = df.astype({"Date": int})
        source = df.loc[(df["Date"] == int(date)) & (df["Pulse number"] == exp_nr)][
            "ITTE (Hz)"
        ]
        self.all_files_info_df.loc[
            (self.all_files_info_df["date"] == "230215")
            & (self.all_files_info_df["discharge_nr"] == 61),
            "frequency",
        ] = 200

        for index, row in self.all_files_info_df.iterrows():
            filtered_df = df.loc[
                (df["Date"] == int(f"20{row['date']}"))
                & (df["Pulse number"] == row["discharge_nr"]),
                "ITTE (Hz)",
            ]

            if not filtered_df.empty:
                wartosc = filtered_df.iloc[0]
                self.all_files_info_df.at[index, "frequency"] = wartosc
            else:
                self.all_files_info_df.at[index, "frequency"] = 200
        print(self.all_files_info_df)

        return df

    def assign_frequency(self):
        self.all_files_info_df["frequency"] = 0
        pass

    def assign_exp_nr(self):
        """
        Assigns the discharge number to each data point in `self.all_files_info_df` based on its `utc_time`.

        The discharge number is taken from `self.triggers_df`. For each row in `self.all_files_info_df`, this function
        iterates through each row in `self.triggers_df` to find a corresponding discharge number, by checking if the
        `utc_time` of the current row in `self.all_files_info_df` falls between the `T0` of the current and previous row
        in `self.triggers_df`.

        If a match is found, the discharge number of the corresponding row in `self.triggers_df` is appended to a list.
        The list is then transformed into a numpy array and used to update the `discharge_nr` column in `self.all_files_info_df`.

        If there is no discharge registered during the day, a message is printed with the date.
        """

        ### dodac warunki
        dic = {}
        for idx_df_total, row_total in self.all_files_info_df.iterrows():
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

        try:
            self.all_files_info_df["discharge_nr"].loc[
                np.array([i for i in dic.keys()])
            ] = np.array([i for i in dic.values()])
        except ValueError:
            print(f"\n{self.date} - no discharges registered during the day!\n")

    def save_file(self):
        destination = pathlib.Path.cwd() / "discharge_numbers"
        destination.mkdir(parents=True, exist_ok=True)
        self.all_files_info_df.to_csv(
            destination / f"{self.element}-{self.date}.csv", sep=","
        )
        print("Experimental numbers saved!")


def get_all_subdirectories(element):
    """
    Retrieves all subdirectories in a given directory.
    """
    path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "__Experimental_data"
        / "data"
        / element
    )
    sub_dirs = [f for f in path.iterdir() if f.is_dir() and f.name[0] != (".")]

    return sub_dirs


if __name__ == "__main__":
    elements = ["C"]  # , "C"]
    for element in elements:
        list_of_directories = get_all_subdirectories(element)

        for dir_ in list_of_directories:
            files = Files(dir_)
            date = files.date
            directory = files.directory
            file_list = files.file_list
            file_sizes = files.file_sizes
            if len(file_list) == 0:
                continue
            t = Triggers(date)
            df = t.df
            exp_ass = ExpAssignment(
                element, directory, date, file_list, file_sizes, df, savefile=True
            )
            # break
