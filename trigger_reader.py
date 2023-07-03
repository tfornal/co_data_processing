from datetime import datetime
import calendar
import requests
import numpy as np
import pandas as pd
import pathlib


class Triggers:

    """Retrieves and processes information about experimental programs from the web API. The API returns information about programs in a JSON format."""

    ARCHIVE_PROGINFO = "http://archive-webapi.ipp-hgw.mpg.de/programs.json?from="

    def __init__(self, date, savefile=True):
        """Initialize the instance of the class.

        Parameters
        ----------
        date : str
            A string in the format "YYMMDD" representing the date for which triggers data will be processed.
        savefile : bool, optional
            Whether the processed data should be saved as a csv file, by default False.
        """
        self.date = str(date)
        self.year, self.month, self.day = self.convert_date(self.date)
        self.beginning_of_the_day, self.end_of_the_day = self._convert_to_UTC()
        self.url = self._get_url()
        self.triggers = self._get_triggers_utc()
        self.df = self._create_df()
        if savefile:
            self.save_file()

    def convert_date(self, date):
        year, month, day = (
            int(date[:4]),
            int(date[4:6]),
            int(date[6:]),
        )
        return year, month, day

    def _convert_to_UTC(self):
        """
        Converts the `date` passed as parameter to UTC timestamps for the start and end of the day.

        Returns
        -------
        tuple
            A tuple of two integers representing the UTC timestamps for the start and end of the day of the `date` passed as parameter.
        """

        beginning_of_the_day = datetime(self.year, self.month, self.day, 0, 0, 0, 0)
        finish_of_the_day = datetime(self.year, self.month, self.day, 23, 59, 59, 0)

        beginning_of_the_day = (
            int(round(calendar.timegm(beginning_of_the_day.timetuple())))
            * 1_000_000_000
            + beginning_of_the_day.microsecond * 1_000
        )

        end_of_the_day = (
            int(round(calendar.timegm(finish_of_the_day.timetuple()))) * 1_000_000_000
            + finish_of_the_day.microsecond * 1_000
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
                response = api_response.json()["programs"][shot]["trigger"]["0"][0]
                triggers["T0"].append(response)
            except (IndexError, TypeError):
                triggers["T0"].append(0)

            try:
                response = api_response.json()["programs"][shot]["trigger"]["1"][0]
                triggers["T1"].append(response)
            except (IndexError, TypeError):
                if response == 0:
                    triggers["T1"].append(0)
                else:
                    triggers["T1"].append(
                        response + 60_000_000_000
                    )  ### dodac 60s od startu ECRH

            try:
                response = api_response.json()["programs"][shot]["trigger"]["6"][0]
                triggers["T6"].append(response)
            except (IndexError, TypeError):
                triggers["T6"].append(0)

        #### backup
        # for shot in range(nr_of_shots):

        #     try:
        #         start_program = api_response.json()["programs"][shot]["trigger"][
        #             "0"
        #         ][0]
        #     except (IndexError, TypeError):
        #         start_program = 0

        #     try:
        #         start_ecrh = api_response.json()["programs"][shot]["trigger"][
        #             "1"
        #         ][0]
        #     except (IndexError, TypeError):
        #         if start_program == 0:
        #             start_ecrh = 0
        #         else:
        #             start_ecrh = start_program + 60_000_000_000 ### dodac 60s od startu ECRH
        #         # start_ecrh = 0 #### sprawdzic wszystkie

        #     try:
        #         end_of_program = api_response.json()["programs"][shot]["trigger"][
        #             "6"
        #         ][0]
        #     except (IndexError, TypeError):
        #         end_of_program = 0
        #     T0.append(start_program)
        #     T1.append(start_ecrh)
        #     T6.append(end_of_program)
        # list_of_discharges = np.arange(1, nr_of_shots + 1)
        # breakpoint()
        return triggers

    def _create_df(self):
        """Creates a pandas DataFrame from the processed triggers data."""
        new_date_fmt = str(self.year) + str(self.month) + str(self.day)
        breakpoint()
        df = pd.DataFrame()

        df["date"] = new_date_fmt
        df["discharge_nr"] = [i for i in range(1, len(self.triggers["T0"]) + 1)]

        df["T0"] = self.triggers["T0"]
        df["T1"] = self.triggers["T1"]
        df["T6"] = self.triggers["T6"]
        breakpoint()
        print(df)
        return df

    # def save_file(self):
    #     destination = pathlib.Path.cwd() / "program_triggers"
    #     destination.mkdir(parents=True, exist_ok=True)
    #     self.df.to_csv(destination / f"{self.date}_triggers.csv")
    #     print("Triggers successfully saved!")


if __name__ == "__main__":
    # date = 20230215
    date = "20230215"
    Triggers(date, savefile=False)
