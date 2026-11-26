import os
import shutil

renamed_folder = r'C:\Users\Danya\Downloads\KIZIM\Renamedscans'

complete_file_set_folder = os.path.join(renamed_folder, 'Complete File Set')
missing_face_folder = os.path.join(renamed_folder, 'Missing Face')
missing_template_folder = os.path.join(renamed_folder, 'Missing Template')
only_mask_and_datasheet_folder = os.path.join(renamed_folder, 'Only Mask and Datasheet')

os.makedirs(complete_file_set_folder, exist_ok=True)
os.makedirs(missing_face_folder, exist_ok=True)
os.makedirs(missing_template_folder, exist_ok=True)
os.makedirs(only_mask_and_datasheet_folder, exist_ok=True)

complete_file_set = [
    '02708', '04030', '04358', '04850', '07922', '08646', '09511', '10456', '12204',
    '12263', '12304', '12306', '12320', '12345', '12380'
]

missing_template = ['11623']

missing_face = [
    '00547', '00617', '02982', '03390', '03725', '04176', '04657', '05582', '06080', '06514',
    '07856', '07920', '08437', '08662', '09037', '09126', '09396', '09449', '09468', '09624',
    '09664', '10076', '10597', '10755', '10809', '10929', '11010', '11148', '11238', '11550',
    '11580', '11594', '11700', '11713', '11716', '11733', '11742', '11748', '11759', '11768',
    '11777', '11782', '11791', '11804', '11805', '12338', '12353', '12524', '12571', '12572',
    '12592', '12593', '12598', '12612', '12614', '12626', '12630', '12635', '12638', '12645',
    '12666', '12685', '12689', '12691', '12696', '12708', '12720', '12721', '12725', '12727',
    '12729', '12780', '12866'
]

only_mask_and_datasheet = ['11521']

def move_files_to_category_folders():
    for filename in os.listdir(renamed_folder):
        if filename.endswith(".stl"):
            patient_id = filename.split('_')[0]

            source_filepath = os.path.join(renamed_folder, filename)

            if patient_id in complete_file_set:
                destination_folder = complete_file_set_folder
            elif patient_id in missing_face:
                destination_folder = missing_face_folder
            elif patient_id in missing_template:
                destination_folder = missing_template_folder
            elif patient_id in only_mask_and_datasheet:
                destination_folder = only_mask_and_datasheet_folder
            else:
                continue

            destination_filepath = os.path.join(destination_folder, filename)
            shutil.move(source_filepath, destination_filepath)
            print(f"Moved {filename} to {destination_folder}")

move_files_to_category_folders()
