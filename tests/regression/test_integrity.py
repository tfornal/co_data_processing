import pytest
from file_reader import FilePathManager
from pathlib import Path


def are_files_equal(file1_path, file2_path):
    with open(file1_path, "rb") as file1, open(file2_path, "rb") as file2:
        return file1.read() == file2.read()


@pytest.fixture
def get_file_paths():
    main_path = Path(__file__).parent.parent.parent.resolve()
    file1 = main_path / "data" / "discharge_numbers" / "C" / "C-20230117.csv"
    file2 = (
        main_path / "tests" / "regression" / "reference_files" / "C-20230117_bac.csv"
    )
    return file1, file2


def test_discharge_nr_files_equality(get_file_paths):
    file1, file2 = get_file_paths
    assert are_files_equal(file1, file2), f"Files are not equal."
