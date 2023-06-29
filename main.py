import time
from reader import get_discharge_nr_from_csv
import pandas as pd
import pathlib

start = time.time()

elements = ["O"]
date = "20230214"
discharges = [i for i in range(100)]
time_interval = [0, 600]


def get_from_csv(element, date):
    filepath = pathlib.Path.cwd() / "discharge_numbers" / f"{element}-{date}.csv"
    df = pd.read_csv(filepath)


if __name__ == "__main__":
    for element in elements:
        for shot in discharges:
            get_discharge_nr_from_csv(element, date, shot, time_interval, plotter=True)

fe = time.time() - start
format_float = "{:.2f}".format(fe)
print(f"Finished within {format_float}s")
