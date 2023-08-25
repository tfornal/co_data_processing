import pandas as pd
import pytest
from file_reader import FilePathManager


@pytest.fixture
def get_test_files():
    path_manager = FilePathManager()
    path = path_manager.program_triggers()
    return [file.name for file in path.glob("*.csv")]


def test_trigger_columns(get_test_files):
    for file in get_test_files:
        path = FilePathManager(None, file).program_triggers() / file
        df = pd.read_csv(path, sep="\t")
        T0 = df["T0"]
        T1 = df["T1"]
        T6 = df["T6"]
        assert (T0 >= 0).all()
        assert (T1 >= 0).all()


def test_sequent_rows(get_test_files):
    for file in get_test_files:
        path = FilePathManager(None, file).program_triggers() / file
        df = pd.read_csv(path, sep="\t")
        array = df.to_numpy()[:, 2:]
        for i, arr_line in enumerate(array):
            if i == 0:
                continue
            x = arr_line - array[i - 1]
            assert x.all() >= 0
