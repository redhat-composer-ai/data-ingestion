from pathlib import Path


def create_data_folders(data_folder_location: str = "./data") -> Path:
    data_folder = Path(data_folder_location)
    raw_folder = data_folder.joinpath("raw")
    processed_folder = data_folder.joinpath("processed")

    raw_folder.mkdir(parents=True, exist_ok=True)
    processed_folder.mkdir(parents=True, exist_ok=True)

    return data_folder


def download_document():
    pass


def convert_pdf_to_markdown():
    pass


if __name__ == "__main__":
    create_data_folders()
