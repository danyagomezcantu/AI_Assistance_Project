import os
import shutil
import re

# Define the source folders
input_folder = r"D:\KI-Assistenz\Initial-Database"  # Folder containing all the files to process
output_folder = r"D:\KI-Assistenz\File-Database"  # Where the renamed files will be stored
unresolved_folder = os.path.join(output_folder, "unresolved")  # Folder for outlier files

# Create output and unresolved folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(unresolved_folder, exist_ok=True)

# Regex pattern to match exactly five consecutive digits followed by a non-digit
five_digit_id_pattern = re.compile(r'(\d{5})(\D)')


# Helper function to find the patient ID in a filename
def extract_patient_id(filename):
    match = five_digit_id_pattern.search(filename)
    if match:
        return match.group(1)  # Return the matched five-digit ID
    return None


# Helper function to rename and copy files
def rename_file(file_path, new_name, output_dir):
    new_path = os.path.join(output_dir, new_name)
    shutil.copy(file_path, new_path)  # Copy the file to the new directory with the new name
    print(f"Copied and renamed file {file_path} to {new_path}")
    return new_path


# Process files
def process_files(input_folder, output_folder, unresolved_folder):
    # List all files in the input folder
    all_files = os.listdir(input_folder)

    for file_name in all_files:
        file_path = os.path.join(input_folder, file_name)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Extract the patient ID (5 consecutive digits followed by a non-digit character)
        patient_id = extract_patient_id(file_name)
        if not patient_id:
            print(f"No valid ID found in {file_name}. Moving to unresolved folder.")
            shutil.move(file_path, os.path.join(unresolved_folder, file_name))
            continue

        # Check the file extension and process only .STL or .PNG files
        if file_name.lower().endswith('.png'):
            new_name = f"{patient_id}_datasheet.png"
            rename_file(file_path, new_name, output_folder)

        elif file_name.lower().endswith('.stl'):
            # Check if it's a template or mask
            if 'a.stl' in file_name.lower():
                new_name = f"{patient_id}_template.stl"
                rename_file(file_path, new_name, output_folder)

            elif 'g.stl' in file_name.lower():
                new_name = f"{patient_id}_mask.stl"
                rename_file(file_path, new_name, output_folder)

            else:
                # Check for face scan format "XXXX_YYYYY_XX_XX" where YYYYY is the patient ID
                if re.search(rf"_{patient_id}_", file_name):
                    new_name = f"{patient_id}_face_scan.stl"
                    rename_file(file_path, new_name, output_folder)
                else:
                    print(f"Unrecognized STL format in {file_name}. Moving to unresolved folder.")
                    shutil.move(file_path, os.path.join(unresolved_folder, file_name))
        else:
            # Any other file type goes to the unresolved folder
            print(f"Unsupported file type {file_name}. Moving to unresolved folder.")
            shutil.move(file_path, os.path.join(unresolved_folder, file_name))


# Run the process
process_files(input_folder, output_folder, unresolved_folder)
