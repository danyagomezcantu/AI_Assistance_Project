import os
from collections import defaultdict

# Define the folder to check for duplicate files
complete_folder = r"D:\KI-Assistenz\File-Database"

# Dictionary to store filenames without extensions
file_names = defaultdict(list)

# Walk through all files in the folder
for root, dirs, files in os.walk(complete_folder):
    for file in files:
        # Extract the filename without extension
        file_name_without_ext = os.path.splitext(file)[0]
        # Save the full file path in the dictionary
        file_names[file_name_without_ext].append(os.path.join(root, file))

# Check for duplicates
duplicates_found = False
print("Checking for duplicate file names (ignoring extensions):")

for file_name, paths in file_names.items():
    if len(paths) > 1:  # If more than one file has the same name (ignoring extension)
        duplicates_found = True
        print(f"\nDuplicate file name found: {file_name}")
        for path in paths:
            print(f" - {path}")

if not duplicates_found:
    print("No duplicate file names found (ignoring extensions).")
else:
    print("\nDuplicate check completed.")
