import csv
import shutil
from pathlib import Path

# === CONFIGURE THESE PATHS IF NEEDED ===
SOURCE_ROOT = Path(r"D:\KI Assistenz\File Repository\1. Training\complete_file_set")
DEST_FACES  = Path(r"D:\KI Assistenz\AI_Assistance_Project\data\raw\faces")
DEST_MASKS  = Path(r"D:\KI Assistenz\AI_Assistance_Project\data\raw\masks")
REPORT_CSV  = Path(r"D:\KI Assistenz\AI_Assistance_Project\copy_report.csv")

# Optional: set to True to do a "dry run" (no actual copies), useful to preview
DRY_RUN = False

# === HELPER FUNCTIONS ===
def find_one_by_suffix(folder: Path, suffix: str):
    """
    Return exactly one file in 'folder' whose name (case-insensitive)
    ends with 'suffix'. If none, return (None, reason). If multiple, return
    (Path, warning) where Path is the first match and warning explains ambiguity.
    """
    suffix_lower = suffix.lower()
    matches = [p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(suffix_lower)]
    if not matches:
        return None, f"NO_MATCH_FOR_{suffix}"
    if len(matches) > 1:
        # pick the first for copying, but warn
        return matches[0], f"MULTIPLE_MATCHES_FOR_{suffix}"
    return matches[0], ""

def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not DRY_RUN:
        shutil.copy2(src, dst)
    return dst

# === MAIN ===
def main():
    if not SOURCE_ROOT.exists() or not SOURCE_ROOT.is_dir():
        print(f"[ERROR] Source root not found or not a directory: {SOURCE_ROOT}")
        return

    DEST_FACES.mkdir(parents=True, exist_ok=True)
    DEST_MASKS.mkdir(parents=True, exist_ok=True)

    subfolders = sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()])
    total_subfolders = len(subfolders)

    copied_faces = 0
    copied_masks = 0
    issues = 0

    report_rows = []
    header = [
        "subfolder",
        "face_scan_found",
        "face_scan_status",
        "face_scan_src",
        "face_scan_dst",
        "mask_found",
        "mask_status",
        "mask_src",
        "mask_dst",
    ]

    for sub in subfolders:
        face_src, face_note = find_one_by_suffix(sub, "_face_scan.stl")
        mask_src, mask_note = find_one_by_suffix(sub, "_mask.stl")

        face_found = bool(face_src)
        mask_found = bool(mask_src)

        face_status = "OK" if face_found and not face_note else ("WARN_" + face_note if face_found else "MISSING")
        mask_status = "OK" if mask_found and not mask_note else ("WARN_" + mask_note if mask_found else "MISSING")

        face_dst = ""
        mask_dst = ""

        # Copy if found
        if face_found:
            dst = safe_copy(face_src, DEST_FACES)
            face_dst = str(dst)
            copied_faces += 1
        if mask_found:
            dst = safe_copy(mask_src, DEST_MASKS)
            mask_dst = str(dst)
            copied_masks += 1

        # Count issues
        if face_status != "OK" or mask_status != "OK":
            issues += 1

        report_rows.append([
            str(sub),
            str(face_found),
            face_status,
            str(face_src) if face_src else "",
            face_dst,
            str(mask_found),
            mask_status,
            str(mask_src) if mask_src else "",
            mask_dst,
        ])

    # Write report
    try:
        REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(report_rows)
    except Exception as e:
        print(f"[WARN] Could not write report CSV: {REPORT_CSV}\n{e}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Subfolders scanned:      {total_subfolders}")
    print(f"Faces copied:            {copied_faces}")
    print(f"Masks copied:            {copied_masks}")
    expected = 366
    if copied_faces != expected or copied_masks != expected:
        print(f"[NOTE] Expected {expected} of each. You currently have:")
        print(f"       Faces: {copied_faces}  |  Masks: {copied_masks}")
        print("       See report for missing/ambiguous cases.")
    print(f"Issues (missing/ambiguous): {issues}")
    print(f"Report saved to: {REPORT_CSV}")
    if DRY_RUN:
        print("\n[DRY RUN] No files were actually copied. Set DRY_RUN=False to perform the copy.")

if __name__ == "__main__":
    main()
