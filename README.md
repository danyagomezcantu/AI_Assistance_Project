# Legacy scripts (July 2024 – October 2025)

This branch keeps the scripts I wrote before the final thesis pipeline (branch `main`, tag `v1.3.6`). They are kept to show how the work evolved. **None of them produced the results reported in the thesis.**

The branch was created in November 2026 from the old `main`, so the commits from July 2024 to April 2025 keep their original dates. The folders whose names start with a date were added in November 2026. Their commit messages give the approximate date of each script, taken from the files' last-modified dates. The code in these folders is unchanged; where I edited a comment, the commit message says so.

## Contents

| Path | Approx. date | What it does |
|---|---|---|
| `Demo/` | Jul–Aug 2024 (original commits) | Aligns a face scan with PCA: the first principal component of the mesh vertices is rotated onto the Y axis, and the bounding-box centre is moved to the origin. It then renames files from `ID.MM.YY.stl` to `NNNN_ID_MM_YY.stl` and sorts them into `YY.MM` folders. It automated two steps we did by hand in CloudCompare ("Bounding box P.C.A. fit" and "Move bounding box center to origin"). |
| `2024-10_repository_organization/` | Oct 2024 | First organization of the AirTec deliveries into one file repository. |
| `renaming_script.py` | Mar 2025 (committed Apr 2025) | Renames AirTec's export names to the repository convention: `NNNNN_g.stl` → `NNNNN_mask.stl`, `NNNNN.MM.YY.stl` → `NNNNN_face_scan.stl`, `NNNNN.MM.YYa.stl` → `NNNNN_template.stl`, `NNNNN.png` → `NNNNN_datasheet.png`. |
| `reporting_script.py` | Mar 2025 (committed Apr 2025) | Counts patient IDs per category and file type in the file repository (`1. Training`, `2. Testing`, `3. Outliers`, `4. Unterkiefer Files`), merges the Unterkiefer spreadsheets and prints the ID lists as LaTeX tables. |
| `new_files_script.py` | Apr 2025 | Ingests a new delivery. It unzips it into `Processing/`, renames the files with `renaming_script.py`, flattens subfolders (adding `_additional` on name collisions) and groups the files into one folder per ID. |
| `2025-08_voxelization/` | Aug–Sep 2025 | Extends `Demo/`: PCA alignment and centring, then voxelization at a fixed 1.5 mm pitch, hole filling and export of the voxel mesh. |
| `2025-09_first_3d_unet/` | Sep 2025 | First learning pipeline: preprocessing of face–mask pairs into voxel grids, a small 3D encoder–decoder with two skip connections, training and inference. |
| `2025-10_copy_faces_and_masks/` | Oct 2025 | Copies every `*_face_scan.stl` and `*_mask.stl` from `1. Training/complete_file_set` into `data/raw/` and writes a CSV report. |

### `2024-10_repository_organization/`

| File | Approx. date | What it does |
|---|---|---|
| `file_renaming_script.py` | 17 Oct 2024 | First version of the renaming rules. It copies renamed files from `Initial-Database` to `File-Database`. Files without a five-digit ID, or with an unknown STL name, are moved (not copied) to `unresolved/`. |
| `data_editing_script.py` | 18 Oct 2024 | Lists the IDs that have all four file types. It also defines functions that swap the labels `mask` and `template` through a temporary `maskaux` name. Those functions were never run: the labels in the repository come from `file_renaming_script.py` (`_g.stl` → mask, `a.stl` → template). |
| `database_organizing_script.py` | 18 Oct 2024 | The first, disabled part sorts each patient into one of 15 categories by which files exist. It then moves the categories into `Training` (complete file set, missing datasheet), `Testing` (missing mask, missing template, only face and mask) and `Outliers` (everything else). The part that runs only prints the IDs per category. |
| `repeated_files_script.py` | 18 Oct 2024 | Reports files that share a name, ignoring the extension. |
| `organizing.py` | 24 Oct 2024 | Moves renamed scans into category folders, using ID lists written into the script (a snapshot of October 2024). |

## How this connects to the thesis pipeline

- The repository layout and file names from these scripts (`1. Training/complete_file_set`, `missing_datasheet`, `NNNNN_face_scan.stl`, `NNNNN_mask.stl`) are what the indexer of the final pipeline (P1) reads.
- `copy_faces_and_masks.py` became step P0.3 (copy only faces and moulds to local disk) and P1 (pair index).
- `Demo/` and `2025-08_voxelization/` became the orientation and voxelization of P3.
- `2025-09_first_3d_unet/` became P3, P5–P6 (3D U-Net) and P9 (face → mould prototype).

## Known issues (documented in November 2026)

These problems are why the final pipeline was rebuilt rather than extended. Each item names what replaced it.

1. **The `*_mask.stl` files are not in the face's coordinate frame.** They are solid positive moulds. In a 20-patient check, they were rotated about 171° and shifted about 660 mm from the face. `2025-08_voxelization/` aligns face and mask separately, each with its own PCA. `2025-09_first_3d_unet/` applies the face's transform to the mask. Neither puts the pair in a common frame. *Final pipeline:* rigid ICP registration of the mould to the face, checked by test A1.
2. **PCA sign ambiguity.** The direction of each principal axis is arbitrary, so an aligned face can come out turned 180°. *Final pipeline:* an anatomical orientation rule plus a sign refinement against the cohort (125 of 367 faces were turned).
3. **The commit message of 23 Jul 2024 says "PCA on the bounding box … longest axis".** The code actually runs PCA on the mesh vertices. Its first component is the direction of largest variance, which is not always the longest side. Old commit messages are left as they are.
4. **No shared voxel grid.** `mesh.voxelized(pitch)` builds each grid from that mesh's own bounds, so the face and mask arrays have different shapes and origins. *Final pipeline:* one fixed 64³ grid per patient (the face box × 1.15), shared by face and mould.
5. **`fill()` does not close open surfaces.** It fills enclosed cavities only, so an open scan stays a thin shell. *Final pipeline:* solid occupancy by ray parity for closed moulds. Open moulds (104 of 367) are excluded.
6. **Model selection on the test split.** `train.py` keeps the checkpoint with the best test Dice, which makes the test score optimistic. *Final pipeline:* 5-fold cross-validation with an inner validation split; each patient is tested once.
7. **`infer.py` returns the mould at the wrong scale and position.** The marching-cubes vertices are in voxel indices and are never multiplied by the pitch or shifted by the grid origin (which was not saved).
8. **Calls that do not exist in trimesh 5.x:** `VoxelGrid.to_mesh`, `VoxelGrid.dilate` and `Trimesh.remove_degenerate_faces` (`2025-08_voxelization/`, `2025-09_first_3d_unet/`).
9. **`rotation_matrix_from_vectors` breaks for parallel or opposite vectors.** It divides by zero; the 2025 version returns the identity for opposite vectors. This is unlikely with real scans. The function follows a Stack Overflow answer (<https://stackoverflow.com/a/59204638>).
10. **`new_files_script.py`: `merge_to_root` never finishes** when a third file has the same name, because `name_additional.ext` is retried forever.
11. **`reporting_script.py`: `map_folders` resets a category** if the same category name appears under two top-level folders, so the earlier IDs would be lost. With the current layout, categories do not repeat.
12. **`2025-08_voxelization/main.py`** defines `run_batch()` but never calls it, and mixes `legacy.` and plain imports.
13. **The renaming rules accept five-digit IDs only**, while grouping accepts five or six digits.
14. Paths are hard-coded Windows paths.
