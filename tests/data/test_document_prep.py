from collections.abc import Generator
from pathlib import Path

import pytest

from data_ingestion.data.document_prep import create_data_folders


@pytest.fixture
def temporary_directory(tmp_path: Path) -> Generator[Path, None, None]:
    """Fixture to provide a temporary directory for testing."""
    return tmp_path


def verify_folders_created(data_folder_location: Path):
    """
    Helper function to verify if the data subfolders folders are created.
    """
    assert (data_folder_location / "raw").exists()
    assert (data_folder_location / "intermediate").exists()
    assert (data_folder_location / "processed").exists()


def test_create_data_folders_when_folders_do_not_exist(temporary_directory: Path):
    """Test create_data_folders when the folders do not yet exist."""
    data_folder_location = temporary_directory / "data"

    # Ensure the folders do not exist initially
    assert not data_folder_location.exists()

    # Call the function
    returned_path = create_data_folders(str(data_folder_location))

    # Check if the function returns the correct path
    assert returned_path == Path(data_folder_location)

    # Verify folders creation
    verify_folders_created(data_folder_location)


def test_create_data_folders_when_folders_already_exist(temporary_directory: Path):
    """Test create_data_folders when the folders already exist."""
    data_folder_location = temporary_directory / "data"
    raw_folder = data_folder_location / "raw"
    intermediate_folder = data_folder_location / "intermediate"
    processed_folder = data_folder_location / "processed"

    # Manually create the folders
    raw_folder.mkdir(parents=True)
    intermediate_folder.mkdir(parents=True)
    processed_folder.mkdir(parents=True)

    # Ensure the folders exist
    verify_folders_created(data_folder_location)

    # Call the function
    returned_path = create_data_folders(str(data_folder_location))

    # Check if the function returns the correct path
    assert returned_path == Path(data_folder_location)

    # Ensure the folders still exist (no errors raised)
    verify_folders_created(data_folder_location)
