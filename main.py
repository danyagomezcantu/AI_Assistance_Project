from file_processing import *
from file_organizing import *

# To use this script, define the paths where your files are
original_folder = Path("C:/Users/Danya/Downloads/KI-Assistenz/KI-Assistenz_Datenbank_Initial")
processed_folder = Path("C:/Users/Danya/Downloads/KI-Assistenz/KI-Assistenz_Datenbank_Processed")

# Ensure the processed folder exists
os.makedirs(processed_folder, exist_ok=True)

# Extract and order files by date
files = list(original_folder.glob("*.stl"))
sorted_files = sorted(files, key=lambda f: (extract_info(f)[2], extract_info(f)[1], extract_info(f)[0]))

# Determine the starting index for numbering in the processed folder
existing_files = list(processed_folder.glob("*.stl"))
if existing_files:
    max_number = max(int(file.stem.split('_')[0]) for file in existing_files)
    start_index = max_number + 1
else:
    start_index = 0

# Process each file, save it to the processed folder, and rename
renamed_files = []
for file in sorted_files:
    processed_file_path = processed_folder / file.name
    print(file.name)
    process_file(file, processed_file_path)
    renamed_files.append((file, processed_file_path))

# Organize the processed files
renamed_files = rename_files(processed_folder, start_index)
organize_files(renamed_files, processed_folder)