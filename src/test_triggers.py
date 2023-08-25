import numpy as np
import pandas as pd
import pytest

from file_reader import FilePathManager


def test_trigger_columns(date):
    path = FilePathManager(None, date).program_triggers() / f"{date}_triggers.csv"
    df = pd.read_csv(path, sep="\t")
    print(df)
    T0 = df["T0"]
    T1 = df["T1"]
    T6 = df["T6"]
    assert T0.all() >= 0
    assert T0.all() >= 0


def test_sequent_rows(date):
    path = FilePathManager(None, date).program_triggers() / f"{date}_triggers.csv"
    df = pd.read_csv(path, sep="\t")
    array = df.to_numpy()[:, 2:]
    for i, arr_line in enumerate(array):
        if i == 0:
            continue
        x = arr_line - array[i - 1]
        assert x.all() > 0


if __name__ == "__main__":
    date = "20230209"
    test_trigger_columns(date)
    test_sequent_rows(date)
