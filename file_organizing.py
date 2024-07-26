import os
import shutil
from pathlib import Path

def extract_info(file):
    """
    Extracts the original ID, month, and year from the filename.

    Parameters:
    file (Path): The file path object.

    Returns:
    tuple: A tuple containing the original ID, month, and year.

    Explanation:
    - Replaces commas and double dots in the filename with single dots.
    - Splits the filename into parts based on dots.
    - Extracts the original ID, month, and year from the parts.
    """
    filename = file.stem.replace(',', '.').replace('..', '.')
    parts = filename.split(".")
    id_original = parts[0]
    month = int(parts[1])
    year = int(parts[2])
    return id_original, month, year

def rename_files(folder_path, start_index):
    """
    Renames files in the folder based on a new naming convention.

    Parameters:
    folder_path (Path): The path to the folder containing the files.
    start_index (int): The starting index for numbering the files.

    Returns:
    list: A list of tuples containing the old and new file paths.

    Explanation:
    - Retrieves all STL files from the folder.
    - Sorts the files based on the date information extracted from their filenames.
    - Renames each file with a new naming convention and collects the old and new file paths.
    """
    files = list(folder_path.glob("*.stl"))
    sorted_files = sorted(files, key=lambda f: (extract_info(f)[2], extract_info(f)[1], extract_info(f)[0]))

    renamed_files = []
    for i, file in enumerate(sorted_files, start=start_index):
        id_original, month, year = extract_info(file)
        new_filename = f"{i:04d}_{id_original}_{month:02d}_{year:02d}.stl"
        new_filepath = folder_path / new_filename
        renamed_files.append((file, new_filepath))
    return renamed_files

def organize_files(renamed_files, folder_path):
    """
    Organizes files into subfolders based on the date extracted from their filenames.

    Parameters:
    renamed_files (list): A list of tuples containing the old and new file paths.
    folder_path (Path): The path to the folder containing the files.

    Explanation:
    - Extracts the month and year from the new filename.
    - Creates a subfolder based on the month and year.
    - Moves the renamed file to the subfolder.
    """
    for old_file, new_filepath in renamed_files:
        # Extract month and year from the new filename
        new_filename = new_filepath.name
        parts = new_filename.split("_")
        month = int(parts[2])
        year = int(parts[3].split(".")[0])

        # Create subfolder for the month and year
        subfolder = folder_path / f"{year:02d}.{month:02d}"
        os.makedirs(subfolder, exist_ok=True)

        # Move the renamed file to the subfolder
        final_filepath = subfolder / new_filename
        shutil.move(str(old_file), final_filepath)