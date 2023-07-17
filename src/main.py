import time
from intensity import get_discharge_nr_from_csv
import pandas as pd
import pathlib

start = time.time()

elements = ["C"]  # , "O"]  # , "O"]
date = "20230215"


# discharges = [i for i in range(100)]
discharges = [21]  # , 18, 21]
time_interval = [0, 1]


if __name__ == "__main__":
    for shot in discharges:
        for element in elements:
            get_discharge_nr_from_csv(element, date, shot, time_interval, plotter=True)


#### na osiach rowniey dodac cyasz lokalne z eksperymentow - czas rozpoczecia i zakonczenia, a takze ewentualnie dla sprawdzenia utc
