import os
from collections import defaultdict

database_folder = r"D:\KI-Assistenz\File-Database"

# Create a dictionary to keep track of patient files
patient_files = defaultdict(lambda: {"face_scan": False, "template": False, "mask": False, "datasheet": False})

# Helper function to extract the patient ID from the renamed file
def extract_patient_id_from_renamed(filename):
    # Assume the format is "ID_type.extension" (e.g., "12345_face_scan.stl")
    return filename.split('_')[0]  # Extracts the part before the first underscore as the patient ID

# Scan through the renamed files and mark which files each patient has
def identify_complete_patients(folder):
    all_files = os.listdir(folder)

    for file_name in all_files:
        file_path = os.path.join(folder, file_name)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Extract the patient ID
        patient_id = extract_patient_id_from_renamed(file_name)

        # Mark which file type the patient has
        if file_name.endswith('face_scan.stl'):
            patient_files[patient_id]['face_scan'] = True
        elif file_name.endswith('template.stl'):
            patient_files[patient_id]['template'] = True
        elif file_name.endswith('mask.stl'):
            patient_files[patient_id]['mask'] = True
        elif file_name.endswith('datasheet.png'):
            patient_files[patient_id]['datasheet'] = True

    # Now identify which patient IDs have all four files
    complete_patients = []
    for patient_id, files in patient_files.items():
        if all(files.values()):  # Check if all file types (face_scan, template, mask, datasheet) are True
            complete_patients.append(patient_id)

    return complete_patients

def print_full_set_patients(folder):
    # Run the process
    complete_patients = identify_complete_patients(folder)

    # Print the list of patient IDs with complete file sets
    print("Patients with complete file sets (4 files):")
    for patient_id in complete_patients:
        print(patient_id)

    # Optionally, print the total count of patients with complete file sets
    print(f"\nTotal number of patients with complete file sets: {len(complete_patients)}")

# Rename all template files to maskaux
def rename_templates_to_maskaux(folder):
    all_files = os.listdir(folder)

    for file_name in all_files:
        file_path = os.path.join(folder, file_name)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Rename templates to maskaux
        if 'template' in file_name:
            new_name = file_name.replace('template', 'maskaux')
            new_path = os.path.join(folder, new_name)
            os.rename(file_path, new_path)
            print(f"Renamed {file_name} to {new_name}")

# Rename all mask files to template
def rename_masks_to_template(folder):
    all_files = os.listdir(folder)

    for file_name in all_files:
        file_path = os.path.join(folder, file_name)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Rename masks to templates
        if 'mask' in file_name and 'maskaux' not in file_name:  # Make sure we don't rename "maskaux" files here
            new_name = file_name.replace('mask', 'template')
            new_path = os.path.join(folder, new_name)
            os.rename(file_path, new_path)
            print(f"Renamed {file_name} to {new_name}")

# Rename all maskaux files to mask
def rename_maskaux_to_mask(folder):
    all_files = os.listdir(folder)

    for file_name in all_files:
        file_path = os.path.join(folder, file_name)

        # Skip directories, only process files
        if os.path.isdir(file_path):
            continue

        # Rename maskaux to mask
        if 'maskaux' in file_name:
            new_name = file_name.replace('maskaux', 'mask')
            new_path = os.path.join(folder, new_name)
            os.rename(file_path, new_path)
            print(f"Renamed {file_name} to {new_name}")

print_full_set_patients(database_folder)
