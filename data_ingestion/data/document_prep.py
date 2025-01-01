from pathlib import Path


def create_data_folders(data_folder_location: str = "./data") -> Path:
    data_folder = Path(data_folder_location)
    folder_names = ["raw", "intermediate", "processed"]

    for folder_name in folder_names:
        folder = data_folder.joinpath(folder_name)
        folder.mkdir(parents=True, exist_ok=True)

    return data_folder


def download_document():
    pass


def convert_pdf_to_markdown():
    pass


if __name__ == "__main__":
    create_data_folders()
