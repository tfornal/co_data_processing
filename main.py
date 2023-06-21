import time
from reader import get_discharge_nr_from_csv

start = time.time()

element = "C"
date = "20230215"
discharges = [32]
time_interval = [0.01, 2]
dt = 5e-3

if __name__ == "__main__":
    for shot in discharges:
        get_discharge_nr_from_csv(element, date, shot, time_interval, dt, plotter=True)

fe = time.time() - start
format_float = "{:.2f}".format(fe)
print(f"Finished within {format_float}s")
