import os
import pandas as pd


def generate_csv_from_pictures(folder_path, output_csv, headers):
    """
    Generates a CSV file with columns based on the provided headers,
    where each row corresponds to a picture in the folder, and all columns are initialized to 0.

    :param folder_path: Path to the folder containing pictures.
    :param output_csv: Path to the output CSV file.
    :param headers: List of header names for the CSV file.
    """
    if not headers:
        raise ValueError("Headers list cannot be empty.")

    picture_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    data = {"picture_name": picture_files}
    for header in headers:
        data[header] = 0

    df = pd.DataFrame(data)

    df.to_csv(output_csv, index=False, encoding="utf-8")
