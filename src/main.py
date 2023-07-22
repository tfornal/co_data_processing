# import time
from intensity import get_discharge_nr_from_csv
import time
from datetime import datetime, timedelta


def generate_dates_list(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    dates_list = []

    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y%m%d")
        dates_list.append(formatted_date)
        current_date += timedelta(days=1)

    return dates_list


start_date_str = "20221213"
end_date_str = "20230405"
dates_list = generate_dates_list(start_date_str, end_date_str)
discharges = [i for i in range(1, 100)]


dates_list = ["20230118"]  # "20230307"

elements = ["C"]  # , "O"]

discharges = [20]
time_interval = [3, 13]  ## tyl jest przodem, przod jest tylem
### gdy mniej niz max dlugosc pliku - ucina poczatek widma - plik binarny tylem do przodu?
### przy czasie 0-3 s czasy sie sypia!!! TODO

if __name__ == "__main__":
    for element in elements:
        for date in dates_list:
            for shot in discharges:
                try:
                    get_discharge_nr_from_csv(
                        element, date, shot, time_interval, plotter=True
                    )
                except FileNotFoundError:
                    continue
