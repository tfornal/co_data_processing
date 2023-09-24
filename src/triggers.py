from datetime import datetime
import calendar
import requests
import pandas as pd
import pathlib

from file_reader import FilePathManager


class Triggers:
    """Retrieves and processes information about experimental programs from the web API. The API returns information about programs in a JSON format."""

    ARCHIVE_PROGINFO = "http://archive-webapi.ipp-hgw.mpg.de/programs.json?from="

    def __init__(self, date: str, savefile=True):
        """Initialize the instance of the class.

        Parameters
        ----------
        date : str
            A string in the format "YYMMDD" representing the date for which triggers data will be processed.
        savefile : bool, optional
            Whether the processed data should be saved as a csv file, by default False.
        """
        self.date = str(date)
        self.trigger_path = FilePathManager().get_directory_for_program_triggers()
        self.triggers_df = self.read_from_file()

        if self.triggers_df is None:
            self.year, self.month, self.day = self.convert_date(self.date)
            self.beginning_of_the_day, self.end_of_the_day = self._convert_to_utc()
            self.url = self._get_url()
            self.triggers = self._get_triggers_utc()
            self.triggers_df = self._create_df()
            if savefile:
                self.save_file()

    def read_from_file(self):
        try:
            triggers_df = pd.read_csv(
                self.trigger_path / f"{self.date}_triggers.csv", sep="\t"
            )
            return triggers_df

        except FileNotFoundError:
            print(f"{self.date} local Trigger file does not exist. Downloading...")
            return None

    def convert_date(self, date: str) -> tuple:
        year, month, day = (
            int(date[:4]),
            int(date[4:6]),
            int(date[6:]),
        )
        return year, month, day

    def _convert_to_utc(self):
        """
        Converts the `date` passed as parameter to UTC timestamps for the start and end of the day.

        Returns
        -------
        tuple
            A tuple of two integers representing the UTC timestamps for the start and end of the day of the `date` passed as parameter.
        """

        beginning_of_the_day = datetime(self.year, self.month, self.day, 0, 0, 0, 0)
        end_of_the_day = datetime(self.year, self.month, self.day, 23, 59, 59, 0)

        beginning_of_the_day = (
            int(round(calendar.timegm(beginning_of_the_day.timetuple())))
            * 1_000_000_000
            + beginning_of_the_day.microsecond * 1_000
        )

        end_of_the_day = (
            int(round(calendar.timegm(end_of_the_day.timetuple()))) * 1_000_000_000
            + end_of_the_day.microsecond * 1_000
        )

        return beginning_of_the_day, end_of_the_day

    def _get_url(self):
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

    def _create_df(self):
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


if __name__ == "__main__":
    date = "20230215"
    df = Triggers(date, savefile=True).triggers_df
