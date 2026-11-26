from legacy.file_processing import *
from file_organizing import *

ORIGINAL = Path("D:/KI Assistenz/File Repository/Test/Original")
PROCESSED = Path("D:/KI Assistenz/File Repository/Test/Processed")
os.makedirs(ORIGINAL, exist_ok=True)
os.makedirs(PROCESSED, exist_ok=True)

# ---- One place to control voxel settings for ALL files ----
#   - Fixed pitch to guarantee identical voxel size for faces & masks
#   - Automatic sizing per-file, set FIXED_PITCH = None
#   - Good starting point for masks/faces: 1.0–2.0 (assuming mm units in STL)
FIXED_PITCH = 1.5          # mm per voxel edge (set None to auto)
TARGET_CELLS_LONGEST = 200 # only used if FIXED_PITCH is None
MIN_PITCH = 0.5            # lower bound if auto
FILL = True                # fill holes for watertightness
DILATE_ITERS = 0           # try 1–2 if shells are too thin after voxelizing

def run_batch():
    stls = sorted(ORIGINAL.glob("*.stl"))
    if not stls:
        print(f"No STL files found in: {ORIGINAL}")
        return

    for src in stls:
        dst = PROCESSED / f"{src.stem}_processed.stl"
        print(f"\nProcessing: {src.name}")
        summary = process_and_voxelize_file(
            input_path=src,
            output_path=dst,
            pitch=FIXED_PITCH,
            target_cells_longest=TARGET_CELLS_LONGEST,
            min_pitch=MIN_PITCH,
            fill=FILL,
            dilate_iters=DILATE_ITERS
        )
        print(
            f"  → saved: {dst.name}\n"
            f"    pitch={summary['pitch']:.3f}, "
            f"faces: {summary['input_faces']} → {summary['output_faces']}"
        )