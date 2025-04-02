import os
import zipfile
import shutil
import re
import importlib.util

spec = importlib.util.spec_from_file_location("renaming_script", "renaming_script.py")
renaming_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renaming_script)

def unzip_to_processing(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped to {extract_to}")

def rename_all_subfolders(processing_path):
    for item in os.listdir(processing_path):
        item_path = os.path.join(processing_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    renaming_script.rename_stl_files(subitem_path)
                else:
                    renaming_script.rename_stl_files(item_path)

def merge_to_root(processing_path):
    for root, dirs, files in os.walk(processing_path, topdown=False):
        if root == processing_path:
            continue  # Skip the root itself
        for file in files:
            src = os.path.join(root, file)
            dst = os.path.join(processing_path, file)
            base, ext = os.path.splitext(dst)
            while os.path.exists(dst):
                dst = f"{base}_additional{ext}"
            shutil.move(src, dst)
        # Once files are moved, remove the empty folder
        if os.path.isdir(root) and not os.listdir(root):
            os.rmdir(root)
    print("All subfolder contents moved to Processing root.")

def group_files_by_id(processing_path):
    for file in os.listdir(processing_path):
        file_path = os.path.join(processing_path, file)
        if os.path.isfile(file_path) and not file.endswith(('.xls', '.xlsx', '.csv')):
            match = re.match(r"(\d{5,6})", file)
            if match:
                id_folder = os.path.join(processing_path, match.group(1))
                os.makedirs(id_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(id_folder, file))
    print("Files grouped by ID.")

def handle_new_files(input_path, is_zip=True, start_from_grouping=False):
    processing_dir = os.path.join(os.path.dirname(input_path), "Processing")

    if start_from_grouping:
        group_files_by_id(input_path)
        return

    if is_zip:
        unzip_to_processing(input_path, processing_dir)
    else:
        processing_dir = input_path

    rename_all_subfolders(processing_dir)
    merge_to_root(processing_dir)
    group_files_by_id(processing_dir)

if __name__ == "__main__":
    handle_new_files("D:\Scanprojekt #9 H+I.zip", is_zip=True)
    # handle_new_files("path/to/unzipped/folder", is_zip=False)
    # handle_new_files("path/to/folder_with_all_files", start_from_grouping=True)
    pass
