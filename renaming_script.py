import os
import re


def rename_stl_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    patterns = {
        re.compile(r"^(\d{5})_g\.stl$"): "{code}_mask.stl",
        re.compile(r"^(\d{5})\.\d{2}\.\d{2}\.stl$"): "{code}_face_scan.stl",
        re.compile(r"^(\d{5})\.\d{2}\.\d{2}a\.stl$"): "{code}_template.stl",
        re.compile(r"^(\d{5})\.png$"): "{code}_datasheet.png"
    }

    try:
        for filename in os.listdir(directory):
            for pattern, new_format in patterns.items():
                match = pattern.match(filename)
                if match:
                    new_filename = new_format.format(code=match.group(1))
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)

                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                    else:
                        print(f"Skipping: {new_filename} already exists.")
                    break  # Exit loop after first match
            else:
                print(f"Skipping: {filename} (does not match any pattern)")
    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    rename_stl_files(folder_path)
