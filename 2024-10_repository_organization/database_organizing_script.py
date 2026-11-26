"""
import os
import shutil
from collections import defaultdict
import csv

# Define the database folder
database_folder = r"D:\KI-Assistenz\File-Database"

# Initialize patient_files dictionary
patient_files = defaultdict(lambda: {'face_scan': False, 'template': False, 'mask': False, 'datasheet': False})

# List of file types
file_types = ['face_scan', 'template', 'mask', 'datasheet']

# Scan the files in the database_folder
for filename in os.listdir(database_folder):
    filepath = os.path.join(database_folder, filename)
    if os.path.isfile(filepath):
        # Extract patient ID and file type
        # Filename format: ID_filetype.extension
        parts = filename.split('_', 1)
        if len(parts) != 2:
            print(f"Filename '{filename}' not in expected format, skipping.")
            continue
        patient_id = parts[0]
        # Check if patient_id is 5 or 6 digits
        if not patient_id.isdigit() or len(patient_id) not in [5, 6]:
            print(f"Patient ID '{patient_id}' is not 5 or 6 digits, skipping.")
            continue
        filetype_with_ext = parts[1].lower()
        filetype = os.path.splitext(filetype_with_ext)[0]
        if filetype in ['face_scan', 'template', 'mask', 'datasheet']:
            patient_files[patient_id][filetype] = filepath
        else:
            print(f"Unknown file type '{filetype}' in filename '{filename}', skipping.")
            continue

# Mapping from file presence to folder names
folder_mapping = {
    (True, True, True, True): 'Complete file set',
    (False, True, True, True): 'Missing face',
    (True, False, True, True): 'Missing template',
    (True, True, False, True): 'Missing mask',
    (True, True, True, False): 'Missing datasheet',
    (True, True, False, False): 'Only face and template',
    (True, False, True, False): 'Only face and mask',
    (True, False, False, True): 'Only face and datasheet',
    (False, True, True, False): 'Only template and mask',
    (False, True, False, True): 'Only template and datasheet',
    (False, False, True, True): 'Only mask and datasheet',
    (True, False, False, False): 'Only face',
    (False, True, False, False): 'Only template',
    (False, False, True, False): 'Only mask',
    (False, False, False, True): 'Only datasheet'
}

# Create folders inside the database_folder
for folder in folder_mapping.values():
    folder_path = os.path.join(database_folder, folder)
    os.makedirs(folder_path, exist_ok=True)

# Prepare CSV file
csv_file_path = os.path.join(database_folder, 'patient_files.csv')
csv_columns = ['Patient ID', 'face_scan', 'template', 'mask', 'datasheet']

folder_patients = defaultdict(list)  # For printing IDs in each folder

with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=csv_columns)
    csvwriter.writeheader()

    for patient_id, files in patient_files.items():
        # Determine which files the patient has
        face = bool(files['face_scan'])
        template = bool(files['template'])
        mask = bool(files['mask'])
        datasheet = bool(files['datasheet'])
        # Build the key for folder mapping
        file_presence = (face, template, mask, datasheet)
        folder_name = folder_mapping.get(file_presence)
        if not folder_name:
            print(f"Patient {patient_id} has an unrecognized file combination: {file_presence}")
            continue
        folder_path = os.path.join(database_folder, folder_name)
        # Move the files to the folder
        for filetype, filepath in files.items():
            if filepath:
                filename = os.path.basename(filepath)
                destination = os.path.join(folder_path, filename)
                shutil.move(filepath, destination)
        # Add patient ID to folder_patients
        folder_patients[folder_name].append(patient_id)
        # Write to CSV
        csv_row = {
            'Patient ID': patient_id,
            'face_scan': 'x' if face else '',
            'template': 'x' if template else '',
            'mask': 'x' if mask else '',
            'datasheet': 'x' if datasheet else ''
        }
        csvwriter.writerow(csv_row)

# Print IDs in each folder
print("\nPatient IDs in each folder:")
for folder_name, patient_ids in folder_patients.items():
    print(f"\nFolder '{folder_name}' contains {len(patient_ids)} patients:")
    print(', '.join(patient_ids))

# Create the three bigger folders
training_folder = os.path.join(database_folder, 'Training')
testing_folder = os.path.join(database_folder, 'Testing')
outliers_folder = os.path.join(database_folder, 'Outliers')

os.makedirs(training_folder, exist_ok=True)
os.makedirs(testing_folder, exist_ok=True)
os.makedirs(outliers_folder, exist_ok=True)

# Move specified folders into the bigger folders
# Training: move "Complete file set" and "Missing datasheet"
training_folders = ['Complete file set', 'Missing datasheet']
for folder_name in training_folders:
    source_folder = os.path.join(database_folder, folder_name)
    if os.path.exists(source_folder):
        shutil.move(source_folder, training_folder)

# Testing: move "Missing mask", "Missing template", "Only face and mask"
testing_folders = ['Missing mask', 'Missing template', 'Only face and mask']
for folder_name in testing_folders:
    source_folder = os.path.join(database_folder, folder_name)
    if os.path.exists(source_folder):
        shutil.move(source_folder, testing_folder)

# Outliers: move any other folders here
for folder_name in folder_mapping.values():
    source_folder = os.path.join(database_folder, folder_name)
    if os.path.exists(source_folder):
        shutil.move(source_folder, outliers_folder)

print("\nFiles have been organized into the appropriate folders.")
"""

import os

# Define the database folder where categories are located
database_folder = r"D:\KI-Assistenz\File-Database"

# Define the names of the categories
categories = ['Training', 'Testing', 'Outliers']

# Function to extract patient ID from the filename (assuming filename format: ID_filetype.extension)
def extract_patient_id(filename):
    return filename.split('_')[0]

# Traverse the categories and subcategories, and print the patient IDs and their counts
for category in categories:
    category_path = os.path.join(database_folder, category)
    if os.path.exists(category_path):
        print(f"\nCategory: {category}")

        total_patient_ids = set()  # Track total unique IDs for the category

        # Iterate over subcategories within each category
        for subcategory in os.listdir(category_path):
            subcategory_path = os.path.join(category_path, subcategory)

            if os.path.isdir(subcategory_path):
                patient_ids = set()  # Use a set to store unique patient IDs

                # Collect patient IDs from files in the subcategory
                for filename in os.listdir(subcategory_path):
                    if os.path.isfile(os.path.join(subcategory_path, filename)):
                        patient_id = extract_patient_id(filename)
                        patient_ids.add(patient_id)
                        total_patient_ids.add(patient_id)

                # Print the subcategory name, the IDs it contains, and the count
                print(f"-- Subcategory: {subcategory} (Total IDs: {len(patient_ids)})")
                if patient_ids:
                    print(f"   Patient IDs: {', '.join(sorted(patient_ids))}")
                else:
                    print(f"   No patient files found.")

        # Print the total count of unique patient IDs in the category
        print(f"Total unique IDs in '{category}': {len(total_patient_ids)}")
    else:
        print(f"\nCategory '{category}' does not exist.")

print("\nFolder check completed.")
