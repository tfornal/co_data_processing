## napisac proceduralnie
"""
The code must contain only the data folder containing the corresponding folders 
in "YYMMDD" format. Subsequent sub-folders (with their names in YYMMDD format) 
must contain the data recorded by the C/O monitor system (in *.dat format).
Needs to be executed only once.
"""
import calendar
from pathlib import Path, PosixPath
from dateutil import tz
from datetime import datetime
from itertools import product

import natsort
import numpy as np
import pandas as pd

from file_reader import FileInformationCollector, FilePathManager
from triggers import Triggers

pd.options.mode.chained_assignment = None

TIME_AFTER_DISCHARGE = 15e9


class ArgsToAcquisitionParametersDataFrame:
    def __init__(self, element, path):
        self.element = element
        self.path = path
        self.date = self._extract_date_from_path(self.path)
        self.triggers_df = self._get_triggers(self.date)
        self.files_list = self._get_files_list(self.element, self.date)
        self.nr_of_frames = self._get_nr_of_frames(self.element, self.date)
        self.camera_setups = self._get_setup_notes(self.element)

    def _get_exp_data_parameters_directory(self):
        fpm = FilePathManager()
        exp_data_parameters_dir_path = fpm.get_directory_for_exp_data_parameters()
        return exp_data_parameters_dir_path

    def get_exp_data_files_directory(self, element: str, date: str) -> PosixPath:
        fpm = FilePathManager()
        main_directory_for_exp_files = fpm.get_directory_for_exp_data(element, date)
        return main_directory_for_exp_files

    def _get_setup_notes(self, element):
        path_exp_data_params = self._get_exp_data_parameters_directory()
        setup_notes = path_exp_data_params / f"{element}-camera_setups.csv"
        with open(setup_notes, "r") as data:
            camera_setups = pd.read_csv(
                data, sep=",", usecols=["date", "discharge_nr", "ITTE_frequency"]
            )
            camera_setups = camera_setups.astype({"date": int})

        # both dataframes indexed by "date" and "discharge_nr" columns
        camera_setups = camera_setups.set_index(["date", "discharge_nr"])
        return camera_setups

    def _extract_date_from_path(self, dir_path: str) -> str:
        date = dir_path.stem
        return date

    def _get_triggers(self, date: str) -> pd.DataFrame:
        triggers_df = Triggers().grab_triggers_df(date)
        return triggers_df

    def _get_det_size(self, binary_file):
        # nr of rows is the number of collected frames
        binary_file.seek(4)
        bites = binary_file.read(4)
        nrows = int.from_bytes(bites, "little")
        return nrows

    def _get_list_of_files_paths(self, element: str, date: str):
        main_directory_for_exp_files = self.get_exp_data_files_directory(element, date)
        files_paths = list(main_directory_for_exp_files.glob("**/*"))
        return files_paths

    def _get_files_list(self, element: str, date: str):
        files_paths = self._get_list_of_files_paths(element, date)
        files_list = [x.stem for x in files_paths if x.is_file()]
        return files_list

    def _get_nr_of_frames(self, element: str, date: str):
        files_paths = self._get_list_of_files_paths(element, date)
        nr_of_frames = []
        for file_path in files_paths:
            with open(file_path, "rb") as binary_file:
                rows_number = self._get_det_size(binary_file)
                nr_of_frames.append(rows_number)
        return nr_of_frames


class AcquisitionParametersFinder(ArgsToAcquisitionParametersDataFrame):
    def __init__(self, element, path, savefile=True):
        super().__init__(element, path)
        self.files_info_df = self._make_df(self.path)
        self.files_info_df = self._calculate_utc_ns_time(self.files_info_df)
        # self.files_info_df = self._get_frequency(
        #     self.files_info_df,
        #     self.camera_setups,
        #     self.files_list,
        #     self.nr_of_frames,
        # )

        self.exp_setup_df = self._merge_exp_setups_dataframes(
            self.files_info_df, self.camera_setups
        )
        self.files_info_df = self._assign_frequency(self.exp_setup_df)
        self.recorded_frames_amount = self._extract_frames_nr(
            self.files_list, self.nr_of_frames
        )
        self.files_info_df = self._merge_finfo_frames_nr_dataframes(
            self.files_info_df, self.recorded_frames_amount
        )

        self.files_info_df = self._update_acquisition_time(self.files_info_df)
        self.files_info_df = self._calc_utc_start_time_ns(self.files_info_df)
        self.files_info_df = self._assign_discharge_nrs(
            self.files_info_df,
            self.triggers_df,
        )

        self.files_info_df = self._shift_to_T1(
            self.files_info_df,
            self.triggers_df,
        )
        if savefile:
            self.save_to_file(self.element, self.date)

    def _get_exp_numbers_directory(self, element: str) -> Path:
        fpm = FilePathManager()
        exp_numbers_dir_path = fpm.get_directory_for_exp_numbers(element)
        return exp_numbers_dir_path

    def _get_file_names(self, path: PosixPath) -> Path:
        fic = FileInformationCollector()
        return fic.get_files_names(path)

    def _get_file_sizes(self, path: PosixPath) -> list:
        fic = FileInformationCollector()
        return fic.get_files_sizes(path)

    def _convert_fname_to_time(self, file_name):
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

    def _get_info_from_fname(self, path):
        fnames_list = self._get_file_names(path)
        return list(
            map(lambda file_name: self._convert_fname_to_time(file_name), fnames_list)
        )

    def _check_if_longer(self, list_of_sublists: list) -> bool:
        any_longer_than_3 = any(len(sublist) > 3 for sublist in list_of_sublists)

        if any_longer_than_3:
            return True

    def _make_df(self, path) -> pd.DataFrame:
        fnames_list = self._get_file_names(path)
        files_sizes = self._get_file_sizes(path)
        splitted_fnames = self._get_info_from_fname(path)
        columns = ["date", "time", "miliseconds"]
        if self._check_if_longer(splitted_fnames):
            columns.append("type_of_data")
            df = pd.DataFrame(splitted_fnames, columns=columns)
        else:
            df = pd.DataFrame(splitted_fnames, columns=columns)
            df["type_of_data"] = "spectrum"

        df["file_name"] = fnames_list
        df["file_size"] = files_sizes
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

    def _convert_str_time_to_utc_datetime(
        self, date: str, time: str, miliseconds: str
    ) -> datetime:
        from_zone = tz.gettz("Europe/Berlin")
        to_zone = tz.gettz("UTC")
        utc_datetime = (
            datetime.strptime(f"{date} {time} {miliseconds}", "%Y%m%d %H%M%S %f")
            .replace(tzinfo=from_zone)
            .astimezone(to_zone)
        )
        return utc_datetime

    def _mili_to_nano(self, miliseconds: str) -> int:
        return int(miliseconds) * 1e6

    def _convert_human_to_utc_ns(self, date: str, time: str, miliseconds: str) -> int:
        utc_datetime = self._convert_str_time_to_utc_datetime(date, time, miliseconds)
        nanoseconds = self._mili_to_nano(miliseconds)
        utc_time_in_ns = int(
            round(calendar.timegm(utc_datetime.utctimetuple())) * 1e9 + nanoseconds
        )
        return utc_time_in_ns

    def _calculate_utc_ns_time(self, df: pd.DataFrame):
        df["utc_time"] = df.apply(
            lambda row: self._convert_human_to_utc_ns(
                row["date"], row["time"], row["miliseconds"]
            ),
            axis=1,
        )
        df["utc_time"] = df["utc_time"].astype("int64")

        return df

    def _merge_exp_setups_dataframes(self, files_info_df: pd.DataFrame, camera_setups):
        """Merges two dataframes based on the dates and discharge numbers."

        Args:
            files_info_df (pd.DataFrame): _description_
            camera_setups (_type_): _description_

        Returns:
            _type_: _description_
        """
        files_info_df["discharge_nr"] = "-"
        files_info_df = files_info_df.set_index(["date", "discharge_nr"])
        # merge both dataframes by indexed columns (date, discharge_nr)
        merged_df = files_info_df.merge(
            camera_setups, left_index=True, right_index=True, how="left"
        )
        return merged_df

    def _assign_frequency(self, merged_df):
        merged_df["frequency"] = merged_df["ITTE_frequency"]
        merged_df.drop("ITTE_frequency", axis=1, inplace=True)
        merged_df["frequency"].fillna(200, inplace=True)
        files_info_df = merged_df.sort_values(by="file_name")
        files_info_df["frequency"] = files_info_df["frequency"].astype(int)

        return files_info_df

    def _extract_frames_nr(self, file_list, nr_of_frames):
        recorded_frames_amount = pd.DataFrame(
            {"file_name": file_list, "frames_amount": nr_of_frames}
        )
        return recorded_frames_amount

    def _merge_finfo_frames_nr_dataframes(self, files_info_df, recorded_frames_amount):
        files_info_df = files_info_df.reset_index().merge(
            recorded_frames_amount, on="file_name"
        )
        return files_info_df

    def _update_acquisition_time(self, files_info_df):
        dt = 1 / files_info_df["frequency"]
        time = files_info_df["frames_amount"] * dt
        files_info_df["acquisition_time"] = time.round(2)

        return files_info_df

    def _calc_utc_start_time_ns(self, files_info_df):
        files_info_df["utc_start_time"] = files_info_df["utc_time"] - (
            1e9 * files_info_df["acquisition_time"]
        ).astype("int64")
        return files_info_df

    def _grab_discharge_nrs(self, files_info_df, triggers_df):
        """Compares two dataframes: the one containing information about
        all the files at a given date and the triggers df containing information
        about all the triggers that occured that day.
        """
        assigned_indexes = {}
        for idx_df_total, row_total in files_info_df.iterrows():
            utc_time = row_total["utc_time"]
            for idx_df_triggers, row_triggers in triggers_df.iterrows():
                utc_time = row_total["utc_time"]
                is_large_file = row_total["file_size"] > 10
                is_bgr_file = "BGR" in row_total["file_name"]
                utc_start_time = row_total["utc_start_time"]
                t1 = row_triggers["T1"]
                t6 = row_triggers["T6"]
                discharge_nr_trigger = row_triggers["discharge_nr"]
                if (
                    (
                        not is_bgr_file
                        and idx_df_triggers > 0
                        and is_large_file
                        and (
                            # TIME_AFTER_DISCHARGE involves data that was
                            # saved after trigger T6 (just in case
                            # saving to a file took too much time)
                            (t1 < utc_time < t6 + TIME_AFTER_DISCHARGE)
                            or (t1 < utc_start_time < utc_time and t1 < utc_time < t6)
                        )
                    )
                    or is_bgr_file
                    and (t6 > utc_time)
                ):
                    assigned_indexes[idx_df_total] = discharge_nr_trigger
                    break

        return assigned_indexes

    def _assign_discharge_nrs(
        self, files_info_df: pd.DataFrame, triggers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """TODO

        Args:
            files_info_df (pd.DataFrame):
            triggers_df (pd.DataFrame):

        Returns:
            pd.DataFrame: Updated dataframe with assigned discharge numbers.
        """
        assigned_indexes = self._grab_discharge_nrs(files_info_df, triggers_df)
        if assigned_indexes:
            files_info_df["discharge_nr"].loc[
                np.array(list(assigned_indexes.keys()))
            ] = np.array(list(assigned_indexes.values()))
            indexes_to_assign = files_info_df.index[
                ~files_info_df.index.isin(assigned_indexes)
            ]
            files_info_df["type_of_data"].loc[indexes_to_assign] = "trash"
        else:
            date = files_info_df.loc[0, "date"]
            print(f"\n{date} - no discharges registered during the day!\n")

        files_info_df["discharge_nr"] = files_info_df["discharge_nr"].astype(
            {"discharge_nr": "int32"}, errors="ignore"
        )
        return files_info_df

    def _shift_to_T1(self, files_info_df, triggers_df):
        triggers_df = triggers_df[["discharge_nr", "T1"]]
        merged_df = pd.merge(files_info_df, triggers_df, on="discharge_nr", how="left")
        filtered_df = merged_df
        filtered_df["new_time"] = (
            filtered_df["utc_start_time"]
            - filtered_df["utc_start_time"]
            + filtered_df["T1"]
            + filtered_df["acquisition_time"] * 1e9
        )
        filtered_df["new_time"] = filtered_df["new_time"].fillna(0).astype(int)
        filtered_df["offset"] = filtered_df["utc_time"] - filtered_df["new_time"]
        files_info_df = filtered_df.drop("T1", axis=1)

        def test():
            neighboring_discharge_indexes = self._extract_neighboring_discharge_indexes(
                files_info_df
            )

            selected = files_info_df.loc[neighboring_discharge_indexes.keys()]
            selected = selected[~selected["type_of_data"].str.contains("BGR")]
            if "20230215" in self.date:
                breakpoint()
            ### selected ["utc_time"] przesunac
            ### utc_time - czas zapisu pliku przez komputer
            ### utc_start_time - obliczony czas rozpoczecia zapisu przed skorygowaniem o offset
            ### new_time - czas zakonczenia akwizycji danego pliku (uwzglednia offset) pomiedzy wystartowaniem a T1

            first_offset = selected.groupby("discharge_nr")["offset"].transform("first")

            selected["time_after_offset"] = first_offset
            first_offset = (
                selected["utc_start_time"]
                - selected["time_after_offset"]
                + selected["acquisition_time"] * 1e9
            )
            files_info_df.loc[first_offset.index, "new_time"] = first_offset.values

            if "20230215" in self.date:
                breakpoint()
            return files_info_df

        files_info_df = test()

        return files_info_df

    def _extract_neighboring_discharge_indexes(self, df):
        discharges_list = df["discharge_nr"].to_list()
        multiple_discharge_files = {}
        for i in range(1, len(discharges_list)):
            if (
                discharges_list[i] == discharges_list[i - 1]
                and discharges_list[i] != "-"
            ):
                multiple_discharge_files[i - 1] = discharges_list[i]
                multiple_discharge_files[i] = discharges_list[i]

        return multiple_discharge_files

    # ##########BACKUP
    # def _shift_to_T1(self, files_info_df, triggers_df):
    #     calibrated_start_times = {}
    #     discharge_nr = 0
    #     file_save_time_offset = 0
    #     for idx_df_total, row_total in files_info_df.iterrows():
    #         for idx_df_triggers, row_triggers in triggers_df.iterrows():
    #             if (
    #                 row_total["discharge_nr"] == row_triggers["discharge_nr"]
    #                 and "BGR" not in row_total["file_name"]
    #             ):
    #                 file_save_time_offset = (
    #                     row_total["utc_start_time"] - row_triggers["T1"]
    #                 )
    #                 if row_total["discharge_nr"] != discharge_nr:
    #                     # file_save_time_offset = offset
    #                     discharge_nr = row_total["discharge_nr"]
    #                 calibrated_start_times[idx_df_total] = file_save_time_offset
    #     files_info_df["new_time"] = [0] * len(files_info_df)
    #     indexes = list(calibrated_start_times.keys())
    #     file_save_time_offset = list(calibrated_start_times.values())
    #     files_info_df["new_time"].loc[indexes] = (
    #         files_info_df["utc_start_time"].loc[indexes]
    #         - file_save_time_offset
    #         + files_info_df["acquisition_time"] * 1e9
    #     )
    #     ## new_time - jest to czas zapisu pliku po uwzglednieniu offsetu
    #     ## (offset = T1 - (czas zapisu - duration))
    #     files_info_df["new_time"] = files_info_df["new_time"].astype("int64")
    #     return files_info_df

    ##########BACKUP

    def _create_directory(self, element):
        path = self._get_exp_numbers_directory(element)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_to_file(self, element, date):
        path = self._create_directory(element)
        self.files_info_df.to_csv(path / f"{element}-{date}.csv", sep="\t")
        print(f"{element}_{date} - Discharge numbers and other details saved!")


def get_exp_data_subdirs(element):
    """
    Retrieves all subdirectories in a given directory.
    """
    path = Path(__file__).parent.parent.resolve() / "data" / "exp_data" / element
    sub_dirs = [f for f in path.iterdir() if f.is_dir() and f.name[0] != (".")]

    return natsort.os_sorted(sub_dirs)


if __name__ == "__main__":
    elements = ["C", "O"]  # , "O"]
    for element in elements:
        list_of_dirs = get_exp_data_subdirs(element)
        for dir_path in list_of_dirs:
            # if "20230117" in str(directory):
            try:
                exp_ass = AcquisitionParametersFinder(element, dir_path, savefile=True)
            except ValueError:
                print(f" {dir_path} - Cannot process the data - continuing.")
                continue
