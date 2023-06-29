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

        self.date = date
        self.beginning_of_the_day, self.end_of_the_day = self._convert_to_UTC()
        self.url = self._get_url()
        self.list_of_discharges, self.T0, self.T1, self.T6 = self._get_triggers_utc()
        self.df = self._create_df()
        if savefile:
            self.save_file()

    def _convert_to_UTC(self):
        """
        Converts the `date` passed as parameter to UTC timestamps for the start and end of the day.

        Returns
        -------
        tuple
            A tuple of two integers representing the UTC timestamps for the start and end of the day of the `date` passed as parameter.
        """

        year, month, day = (
            int(self.date[:4]),
            int(self.date[4:6]),
            int(self.date[6:]),
        )
        # breakpoint()
        beginning_of_the_day = datetime(year, month, day, 0, 0, 0, 0)
        finish_of_the_day = datetime(year, month, day, 23, 59, 59, 0)

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

        tab_startow = []
        tab_konca = []
        T0 = []
        T1 = []
        T6 = []

        wywolanie_dnia = requests.get(self.url)
        number_of_discharges = len(wywolanie_dnia.json()["programs"])
        print(f"Number of discharges on 20{self.date}:", number_of_discharges)

        for discharge_nr in range(number_of_discharges):
            pole_poczatek = wywolanie_dnia.json()["programs"][discharge_nr]["from"]
            tab_startow.append(pole_poczatek)
            pole_koniec = wywolanie_dnia.json()["programs"][discharge_nr]["upto"]
            tab_konca.append(pole_koniec)
            try:
                start_program = wywolanie_dnia.json()["programs"][discharge_nr][
                    "trigger"
                ]["0"][0]
            except (IndexError, TypeError):
                start_program = 0

            try:
                start_ecrh = wywolanie_dnia.json()["programs"][discharge_nr]["trigger"][
                    "1"
                ][0]
            except (IndexError, TypeError):
                if start_program == 0:
                    start_ecrh = 0
                else:
                    start_ecrh = (
                        start_program + 60_000_000_000
                    )  ### dodac 60s od startu ECRH
                # start_ecrh = 0 #### sprawdzic wszystkie

            try:
                end_of_program = wywolanie_dnia.json()["programs"][discharge_nr][
                    "trigger"
                ]["6"][0]
            except (IndexError, TypeError):
                end_of_program = 0
            T0.append(start_program)
            T1.append(start_ecrh)
            T6.append(end_of_program)
        list_of_discharges = np.arange(1, number_of_discharges + 1)
        return list_of_discharges, T0, T1, T6

    def _create_df(self):
        """Creates a pandas DataFrame from the processed triggers data.

        Returns
        -------
        _type_
            _description_
        """
        new_date_fmt = self.date[:2] + self.date[2:4] + self.date[4:]
        df = pd.DataFrame(columns=["date", "discharge_nr", "T0"])
        df["discharge_nr"] = self.list_of_discharges.astype(int)
        df["T0"] = self.T0
        df["T1"] = self.T1
        df["T6"] = self.T6
        df["date"] = new_date_fmt
        return df

    def save_file(self):
        """Saves the pandas DataFrame as a csv file with the format "YYMMDD_triggers.csv"."""
        destination = pathlib.Path.cwd() / "program_triggers"
        destination.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(destination / f"{self.date}_triggers.csv")
        print("Triggers successfully saved!")
