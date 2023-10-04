from datetime import datetime
import calendar
import requests
import pandas as pd
import pathlib

from file_reader import FilePathManager


# 1 - check if file exists
# 2 - if not - establish connection and download from api
# 3 - get triggers into dataframe; 



class TriggersFromFile:
    @classmethod
    def _get_program_triggers_directory(cls):
        fpm = FilePathManager()
        trigger_directory_path = fpm.get_directory_for_program_triggers()
        return trigger_directory_path

    def _create_trigger_file_path(self, date: str) -> pathlib.Path:
        trigger_directory_path = self._get_program_triggers_directory()
        trigger_path = trigger_directory_path.joinpath(f"{date}_triggers.csv")
        return trigger_path

    def read_from_file(self, date):
        path = self._create_trigger_file_path(date)
        try:
            triggers_df = pd.read_csv(path, sep="\t")
            return triggers_df

        except FileNotFoundError:
            print(f"{date} local Trigger file does not exist. Downloading...")
            return None



class TriggersFromHTTP:

    ARCHIVE_PROGINFO = "http://archive-webapi.ipp-hgw.mpg.de/programs.json?from="

    # def __init__(self):
        
        # if not self.triggers_df:
        # self.beginning_of_the_day, self.end_of_the_day = self._convert_to_utc()
        # self.url = self._get_url()
        # self.triggers = self._get_triggers_utc()
        # self.triggers_df = self._create_df()
        # if savefile:
        #     self.save_file()

    @classmethod
    def _convert_date(cls, date: str) -> tuple:
        try:
            date_obj = datetime.strptime(date, "%Y%m%d")
            year, month, day = date_obj.year, date_obj.month, date_obj.day
            return year, month, day
        except ValueError:
            raise ValueError("Niepoprawny format daty. Oczekiwany format to YYYYMMDD.")
    
    @classmethod
    def _compute_day_start_ns(self, converted_date):
        start_of_the_day = datetime(converted_date*, 0, 0, 0, 0)

        start_time_in_ns = (
            int(round(calendar.timegm(start_of_the_day.timetuple())))
            * 1_000_000_000
            + start_of_the_day.microsecond * 1_000
        )  
        return start_time_in_ns
    
    @classmethod 
    def _compute_day_end_ns(self, converted_date):
        end_of_the_day = datetime(converted_date*, 23, 59, 59, 0)
        end_time_in_ns = (
            int(round(calendar.timegm(end_of_the_day.timetuple())))
            * 1_000_000_000
            + end_of_the_day.microsecond * 1_000
        )  
        return end_time_in_ns

    def _grab_datetime_in_ns(self, converted_date):
        day_start_ns = compute_day_start_ns(converted_date)
        day_end_ns = compute_day_end_ns(converted_date)
        return day_start_ns, day_end_ns

    def _get_url(self, timestamps_in_ns: tuple) -> str:
        """Returns the URL for accessing triggers data for the `date` passed as parameter."""
        url = (
            self.ARCHIVE_PROGINFO
            + str(self.beginning_of_the_day)
            + "&upto="
            + str(self.end_of_the_day)
        )
        return url
    

    def _get_triggers_utc(self):
        """
        Returns the discharge numbers and start of each discharge in UTC timestamps for the day of the `date` passed as parameter.
        """
        triggers = {"T0": [], "T1": [], "T6": []}
        api_response = requests.get(self.url)
        nr_of_shots = len(api_response.json()["programs"])
        print(f"Number of discharges on 20{self.date}:", nr_of_shots)

        for shot in range(nr_of_shots):
            try:
                start_program = api_response.json()["programs"][shot]["trigger"]["0"][0]
                triggers["T0"].append(start_program)
            except (IndexError, TypeError):
                start_program = 0
                triggers["T0"].append(start_program)
            try:
                start_ecrh = api_response.json()["programs"][shot]["trigger"]["1"][0]
                triggers["T1"].append(start_ecrh)
            except (IndexError, TypeError):
                ## TODO - wyprintowac komentarz
                if start_program == 0:
                    start_ecrh = 0
                    triggers["T1"].append(start_ecrh)
                else:
                    start_ecrh = (
                        start_program + 60_000_000_000
                    )  ### add 60s to ECRH start
                    triggers["T1"].append(start_ecrh)
            try:
                end_of_program = api_response.json()["programs"][shot]["trigger"]["6"][
                    0
                ]
                triggers["T6"].append(end_of_program)
            except (IndexError, TypeError):
                end_of_program = start_program + 61_000_000_000
                triggers["T6"].append(end_of_program)
        return triggers


    def get_triggers_df(self):
        """Creates a pandas DataFrame from the processed triggers data."""
        triggers_df = pd.DataFrame()
        triggers_df["discharge_nr"] = [
            i for i in range(1, len(self.triggers["T0"]) + 1)
        ]
        triggers_df["date"] = (
            str(f"{self.year:02d}") + str(f"{self.month:02d}") + str(f"{self.day:02d}")
        )

        for key, value in self.triggers.items():
            triggers_df[f"{key}"] = value
        return triggers_df

    def save_file(self):
        self.trigger_path.mkdir(parents=True, exist_ok=True)
        self.triggers_df.to_csv(
            destination / f"{self.date}_triggers.csv", sep="\t", index=False
        )
        print("Triggers successfully saved!")


class Triggers:
    print("Trrr")

if __name__ == "__main__":
    date = "20230323"
    # triggers_df = TriggersFromFile().read_from_file(date)
    triggers_df = TriggersFromHTTP()
    print(triggers_df)
    # Triggers()
