import pandas as pd
import pytest
from file_reader import FilePathManager
from data_processing import src


@pytest.fixture
def get_trigger_files_paths():
    path_manager = FilePathManager()
    path = path_manager.program_triggers()
    files = [file.name for file in path.glob("*_triggers.csv")]
    # for file in files:
    return [FilePathManager(None, file).program_triggers() / file for file in files]


def test_if_directory_not_empty(get_trigger_files_paths):
    assert len(get_trigger_files_paths) > 0, "No files in a given direcory."


def test_trigger_columns_in_files(get_trigger_files_paths):
    for path in get_trigger_files_paths:
        df = pd.read_csv(path, sep="\t")
        T0 = df["T0"]
        T1 = df["T1"]
        T6 = df["T6"]
        # TODO - zle warunki ponizej
        assert (T0 >= 0).all()
        assert (T1 >= 0).all()


def test_sequent_rows_in_files(get_trigger_files_paths):
    for path in get_trigger_files_paths:
        df = pd.read_csv(path, sep="\t")
        array = df.to_numpy()[:, 2:]
        for i, arr_line in enumerate(array):
            if i == 0:
                continue
            x = arr_line - array[i - 1]
            print(x)
            assert x.all() >= 0


## napisac funkcje, ktora nie przejdzie testow - fakeowy csv
### napisac testy do
