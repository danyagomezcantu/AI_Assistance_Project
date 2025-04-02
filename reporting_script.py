import os
import pandas as pd
import re


def map_folders(directory):
    main_folders = ["1. Training", "2. Testing", "3. Outliers"]
    folder_mapping = {}
    categories = set()

    for main_folder in main_folders:
        main_path = os.path.join(directory, main_folder)
        if not os.path.isdir(main_path):
            print(f"Warning: {main_path} is not a valid directory.")
            continue

        for category in os.listdir(main_path):
            category_path = os.path.join(main_path, category)
            if os.path.isdir(category_path):
                categories.add(category)
                folder_mapping[category] = []

                for patient_id in os.listdir(category_path):
                    patient_path = os.path.join(category_path, patient_id)
                    if os.path.isdir(patient_path) and re.match(r"\d{5,6}$", patient_id):
                        folder_mapping[category].append(patient_id)

    print("Categories Found:")
    for cat in sorted(categories):
        print(f"- {cat}")

    return folder_mapping


def process_unterkiefer(directory):
    unterkiefer_path = os.path.join(directory, "4. Unterkiefer Files")
    if not os.path.isdir(unterkiefer_path):
        print("Warning: Unterkiefer Files directory not found.")
        return [], pd.DataFrame()

    all_files = [f for f in os.listdir(unterkiefer_path) if f.endswith(('.csv', '.xls', '.xlsx'))]
    combined_data = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(unterkiefer_path, file)
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            df = df.drop_duplicates()
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not combined_data.empty:
        combined_data = combined_data.drop_duplicates()
        if 'ID' in combined_data.columns:
            id_list = combined_data['ID'].drop_duplicates().tolist()
        else:
            id_list = []
    else:
        id_list = []

    combined_data.to_csv(os.path.join(unterkiefer_path, "unterkiefer_combined_data.csv"), index=False)

    return id_list, combined_data


def count_unresolved_files(directory):
    unresolved_path = os.path.join(directory, "3. Outliers", "unresolved_files")
    if not os.path.isdir(unresolved_path):
        return 0

    return len([f for f in os.listdir(unresolved_path) if os.path.isfile(os.path.join(unresolved_path, f))])


def total_patient_ids(folder_mapping):
    all_ids = []
    for ids in folder_mapping.values():
        all_ids.extend(ids)
    return sorted(set(all_ids))


def filter_ids_by_filetype(directory, keyword):
    matching_ids = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if keyword in file:
                match = re.search(r"\d{5,6}", root)
                if match:
                    matching_ids.add(match.group())
    return sorted(matching_ids)


def total_face_scans(directory):
    return filter_ids_by_filetype(directory, "face_scan")


def total_masks(directory):
    return filter_ids_by_filetype(directory, "mask")


def total_templates(directory):
    return filter_ids_by_filetype(directory, "template")


def total_datasheet(directory):
    return filter_ids_by_filetype(directory, "datasheet")


def total_additional(directory):
    return filter_ids_by_filetype(directory, "additional")


def format_latex_list(id_list, items_per_row=9):
    id_list = sorted(id_list)
    latex_str = "\\begin{longtable}{|*{9}{>{\\centering\\arraybackslash}p{1.3cm}|}}\\hline\n"

    for i in range(0, len(id_list), items_per_row):
        row = " & ".join(map(str, id_list[i:i + items_per_row]))
        latex_str += row + " \\\ \n"

    latex_str += "\\hline \\end{longtable}"
    return latex_str


def print_category_summaries(folder_mapping):
    categories = [
        "complete_file_set", "face_datasheet_only", "face_mask_only", "face_template_only",
        "mask_datasheet_only", "mask_template_only", "missing_datasheet", "missing_face_scan",
        "missing_mask", "missing_template", "only_datasheet", "only_face", "only_mask",
        "only_template", "template_datasheet_only"
    ]

    for category in categories:
        if category in folder_mapping:
            id_list = folder_mapping[category]
            print(f"\n{category}: {len(id_list)} IDs")
            print(format_latex_list(id_list, 9))


def main():
    directory = r"D:\\KI Assistenz\\File Repository_March2025\\File Repository"
    folder_mapping = map_folders(directory)

    print("\nFolder Mapping:")
    for category, ids in folder_mapping.items():
        print(f"{category}: {ids if ids else 'Empty'}")

    unterkiefer_ids, unterkiefer_data = process_unterkiefer(directory)
    print("\nUnterkiefer IDs:", unterkiefer_ids)

    unresolved_count = count_unresolved_files(directory)
    print(f"\nUnresolved Files Count: {unresolved_count}")

    print("\nTotal Patient IDs:", len(total_patient_ids(folder_mapping)), format_latex_list(total_patient_ids(folder_mapping)))
    print("\nTotal Face Scans:", len(total_face_scans(directory)), format_latex_list(total_face_scans(directory)))
    print("\nTotal Masks:", len(total_masks(directory)), format_latex_list(total_masks(directory)))
    print("\nTotal Templates:", len(total_templates(directory)), format_latex_list(total_templates(directory)))
    print("\nTotal Datasheets:", len(total_datasheet(directory)), format_latex_list(total_datasheet(directory)))
    print("\nTotal Additional:", len(total_additional(directory)), format_latex_list(total_additional(directory)))

    print_category_summaries(folder_mapping)

if __name__ == "__main__":
    main()
