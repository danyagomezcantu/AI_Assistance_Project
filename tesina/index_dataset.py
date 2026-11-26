#!/usr/bin/env python3
"""
index_dataset.py — single source of truth for the KI-Assistenz / AirTec dataset.

WHAT THIS MODULE IS FOR
-----------------------
Everything downstream (preprocessing, cross-validation, training, evaluation)
depends on one question being answered correctly and identically every time:

    "Which patients have BOTH a face scan and a final mask, and which fold
     does each of them belong to?"

This module answers that question once, writes the answer to disk as a
manifest CSV, and never lets anything else re-derive it. If the manifest is
wrong, every number in the dissertation is wrong, so this module is
deliberately paranoid: it verifies rather than assumes, and it halts loudly
when reality disagrees with what we expect.

THE NAMING CONVENTION (not invented here — recovered from the archive)
----------------------------------------------------------------------
The archive was standardised by the author in 2024 with a set of organising /
renaming scripts. Those scripts define the convention this module parses:

    <patient_id>_<kind>.<ext>

    patient_id : 5 or 6 digits, zero-padded (e.g. 00547, 12441)
    kind       : face_scan | template | mask | datasheet
    ext        : stl for the three geometries, png/pdf for the datasheet

Patient folders are themselves named with the bare patient ID.

THE ARCHIVE TAXONOMY (and why we do not trust it blindly)
---------------------------------------------------------
The archive sorts patients into folders by which files they possess:

    1. Training/complete_file_set    366 patients  face+template+mask+datasheet
    1. Training/missing_datasheet      2 patients  face+template+mask, no datasheet
    2. Testing/face_template_only     28 patients  face+template, NO mask
    2. Testing/missing_mask           36 patients  no mask
    3. Outliers/only_face            584 patients  face only
    3. Outliers/only_mask              3 patients  mask only
    4. Unterkiefer Files                           jaw scans — NOT this study

    => complete face+mask pairs = 366 + 2 = 368

That "366 + 2" is the single most important thing in this file. A naive
indexer pointed at `complete_file_set` alone finds 366, fails the 368 check,
and looks broken when it is not. The two extra patients (09751, 12441) have a
face and a mask but no datasheet — and since the datasheet is metadata, not
geometry, they are perfectly valid training pairs.

However: the folder taxonomy is a *claim* made by a 2024 script. The files
actually present on disk are the *evidence*. This module classifies every
patient from its files, then cross-checks that classification against the
folder it was found in, and reports every disagreement. A silent
reclassification would be a data-integrity bug; a reported one is a finding.

WHAT IT EMITS
-------------
    manifest.csv              the 368 complete pairs + fold assignment  [PRIMARY]
    pretrain_faces.csv        face-only patients — encoder-pretraining corpus (RQ9)
    template_validation.csv   patients with a template but no mask — used to
                              test whether a template equals the negative of the
                              final mask (must-verify assumption #2)
    index_report.json         counts, anomalies, taxonomy disagreements

PRIVACY
-------
The manifest contains pseudonymous patient IDs and file paths. It is DATA, not
code. It must never be committed to GitHub. Write it to Drive or to a local
directory covered by .gitignore.

USAGE
-----
    # Colab, archive already unzipped on the local VM disk (fast):
    python index_dataset.py --root /content/KI_Assistenz_File_Repository \
                            --out-dir /content/drive/MyDrive/tesina_outputs/index

    # Colab, straight from the Drive ZIP (extracts first):
    python index_dataset.py --zip "/content/drive/MyDrive/File Repository.zip" \
                            --extract-to /content/KI_Assistenz_File_Repository \
                            --out-dir /content/drive/MyDrive/tesina_outputs/index

    # Validate the logic with no real data at all (do this first):
    python index_dataset.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Convention constants
# --------------------------------------------------------------------------

PATIENT_DIR_RE = re.compile(r"^\d{5,6}$")

FILE_RE = re.compile(
    r"^(?P<pid>\d{5,6})_(?P<kind>face_scan|template|mask|datasheet)\.(?P<ext>[A-Za-z0-9]+)$",
    re.IGNORECASE,
)

GEOMETRY_KINDS = ("face_scan", "template", "mask")
METADATA_KINDS = ("datasheet",)
ALL_KINDS = GEOMETRY_KINDS + METADATA_KINDS

MESH_EXTS = {".stl"}
METADATA_EXTS = {".png", ".pdf"}

#: Top-level archive branches to skip entirely. `4. Unterkiefer Files` holds
#: mandible (lower-jaw) scans from a different sub-project. They are not CPAP
#: masks and must never enter the manifest.
EXCLUDED_BRANCHES = ("4. unterkiefer files",)

#: Folder categories the archive claims contain complete face+mask pairs.
#: Used only for the cross-check — membership is decided by files, not folders.
EXPECTED_PAIR_CATEGORIES = (
    "1. training/complete_file_set",
    "1. training/missing_datasheet",
)

_EXPECTED_LEAVES = {c.split("/")[-1] for c in EXPECTED_PAIR_CATEGORIES}

EXPECTED_COMPLETE_PAIRS = 368


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class PatientRecord:
    """Everything known about one patient folder, derived from its contents."""

    pid: str
    folder: Path
    category: str                      # archive folder, relative, lowercased
    files: dict = field(default_factory=lambda: defaultdict(list))
    unknown_files: list = field(default_factory=list)
    nested_in: str | None = None       # parent patient id, if nested (anomaly)
    id_mismatch: list = field(default_factory=list)  # file pid != folder pid

    # -- presence flags -----------------------------------------------------

    @property
    def has_face(self) -> bool:
        return bool(self.files.get("face_scan"))

    @property
    def has_mask(self) -> bool:
        return bool(self.files.get("mask"))

    @property
    def has_template(self) -> bool:
        return bool(self.files.get("template"))

    @property
    def has_metadata(self) -> bool:
        return bool(self.files.get("datasheet"))

    @property
    def is_complete_pair(self) -> bool:
        """The ONLY definition of 'usable for training' in this project.

        Face + mask. Deliberately does NOT require a datasheet (metadata is
        optional, RQ8) nor a template (derived, not learned).
        """
        return self.has_face and self.has_mask

    # -- chosen file per kind ----------------------------------------------

    def pick(self, kind: str) -> Path | None:
        """Return the single canonical file for `kind`, or None.

        Duplicates are an anomaly, not a crash: we take the
        lexicographically first and flag the count in the manifest so the
        ambiguity is visible in the data rather than hidden in a log.
        """
        candidates = sorted(self.files.get(kind, []))
        return candidates[0] if candidates else None

    def count(self, kind: str) -> int:
        return len(self.files.get(kind, []))


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def _relative_category(folder: Path, root: Path) -> str:
    """Archive category = path of the patient folder's PARENT, relative to root."""
    try:
        rel = folder.parent.relative_to(root)
    except ValueError:
        rel = folder.parent
    return rel.as_posix().lower()


def _is_excluded(folder: Path, root: Path) -> bool:
    try:
        rel = folder.relative_to(root).as_posix().lower()
    except ValueError:
        return False
    return any(rel == b or rel.startswith(b + "/") for b in EXCLUDED_BRANCHES)


def discover_patients(root: Path) -> tuple[list[PatientRecord], list[str]]:
    """Walk the archive and build one PatientRecord per patient folder.

    A patient folder is any directory whose name is a 5-6 digit ID. The walk is
    recursive because the archive contains at least one known nesting anomaly
    (04358 contains 04425); nested patients are indexed as first-class patients
    and flagged, rather than silently absorbed into their parent or dropped.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Archive root not found or not a directory: {root}")

    records: list[PatientRecord] = []
    warnings: list[str] = []
    seen: dict[str, Path] = {}

    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        if not PATIENT_DIR_RE.match(folder.name):
            continue
        if _is_excluded(folder, root):
            continue

        pid = folder.name

        # Nesting anomaly: a patient folder inside another patient folder.
        nested_in = folder.parent.name if PATIENT_DIR_RE.match(folder.parent.name) else None

        if pid in seen:
            warnings.append(
                f"DUPLICATE_PATIENT_FOLDER pid={pid} first={seen[pid]} second={folder} "
                f"(second ignored — resolve manually before trusting the manifest)"
            )
            continue
        seen[pid] = folder

        rec = PatientRecord(
            pid=pid,
            folder=folder,
            category=_relative_category(folder, root),
            nested_in=nested_in,
        )

        for item in sorted(folder.iterdir()):
            if not item.is_file():
                continue
            m = FILE_RE.match(item.name)
            if not m:
                rec.unknown_files.append(item.name)
                continue

            kind = m.group("kind").lower()
            ext = "." + m.group("ext").lower()
            file_pid = m.group("pid")

            # Guard against a mis-filed file from another patient.
            if file_pid != pid:
                rec.id_mismatch.append(item.name)
                continue

            # Guard against a geometry kind carrying a non-mesh extension.
            if kind in GEOMETRY_KINDS and ext not in MESH_EXTS:
                rec.unknown_files.append(item.name)
                continue
            if kind in METADATA_KINDS and ext not in METADATA_EXTS:
                rec.unknown_files.append(item.name)
                continue

            rec.files[kind].append(item)

        records.append(rec)

    return records, warnings


# --------------------------------------------------------------------------
# Deterministic cross-validation folds
# --------------------------------------------------------------------------

def assign_folds(pids: list[str], k: int, seed: int) -> dict[str, int]:
    """Assign each patient to one of k folds, reproducibly.

    Determinism matters more than elegance here. The dissertation claims every
    configuration is compared on IDENTICAL folds ("paired comparisons"), so the
    fold map must be reproducible by another researcher, on another machine, in
    another language, years from now.

    We therefore avoid Python's RNG entirely and derive the ordering from
    SHA-256 of "<seed>:<pid>". This is:
      * stable across Python versions, platforms and library upgrades,
      * independent of the order patients were discovered on disk,
      * trivially re-implementable by a third party for verification.

    Patients are ranked by digest and dealt round-robin, which keeps fold sizes
    balanced to within one patient.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    def digest(pid: str) -> str:
        return hashlib.sha256(f"{seed}:{pid}".encode("utf-8")).hexdigest()

    ordered = sorted(pids, key=digest)
    return {pid: i % k for i, pid in enumerate(ordered)}


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def cross_check_taxonomy(pairs: list[PatientRecord]) -> list[str]:
    """Compare file-derived classification against the archive's folder claim."""
    disagreements = []
    for rec in pairs:
        # Compare by the last folder name, so the check also holds when the ZIP
        # contains only '1. Training' (the walk then starts inside it).
        if rec.category.split("/")[-1] not in _EXPECTED_LEAVES:
            disagreements.append(
                f"UNEXPECTED_SOURCE pid={rec.pid} has face+mask but sits in "
                f"'{rec.category}' (expected one of {list(EXPECTED_PAIR_CATEGORIES)})"
            )
    return disagreements


def build_report(
    records: list[PatientRecord],
    pairs: list[PatientRecord],
    warnings: list[str],
    root: Path,
    k: int,
    seed: int,
) -> dict:
    by_category = Counter(r.category for r in records)
    pair_by_category = Counter(r.category for r in pairs)

    anomalies = {
        "nested_patient_folders": [
            {"pid": r.pid, "nested_in": r.nested_in} for r in records if r.nested_in
        ],
        "duplicate_kind": [
            {"pid": r.pid, "kind": kind, "count": r.count(kind)}
            for r in records
            for kind in ALL_KINDS
            if r.count(kind) > 1
        ],
        "unrecognised_filenames": [
            {"pid": r.pid, "files": r.unknown_files} for r in records if r.unknown_files
        ],
        "patient_id_mismatch": [
            {"pid": r.pid, "files": r.id_mismatch} for r in records if r.id_mismatch
        ],
        "walker_warnings": warnings,
        "taxonomy_disagreements": cross_check_taxonomy(pairs),
    }

    return {
        "archive_root": str(root),
        "totals": {
            "patient_folders_found": len(records),
            "complete_pairs_face_and_mask": len(pairs),
            "face_only": sum(1 for r in records if r.has_face and not r.has_mask),
            "mask_only": sum(1 for r in records if r.has_mask and not r.has_face),
            "with_template": sum(1 for r in records if r.has_template),
            "pairs_with_template": sum(1 for r in pairs if r.has_template),
            "pairs_without_metadata": sum(1 for r in pairs if not r.has_metadata),
            "template_but_no_mask": sum(
                1 for r in records if r.has_template and not r.has_mask
            ),
        },
        "patient_folders_by_category": dict(sorted(by_category.items())),
        "complete_pairs_by_category": dict(sorted(pair_by_category.items())),
        "cross_validation": {"k": k, "seed": seed, "method": "sha256(seed:pid) rank, round-robin"},
        "anomalies": anomalies,
    }


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "pid", "fold", "face_path", "mask_path", "template_path", "datasheet_path",
    "has_template", "has_metadata", "source_category",
    "n_face_files", "n_mask_files", "nested_in", "anomaly_flags",
]


def _anomaly_flags(rec: PatientRecord) -> str:
    flags = []
    if rec.nested_in:
        flags.append("nested")
    if rec.count("face_scan") > 1:
        flags.append("multi_face")
    if rec.count("mask") > 1:
        flags.append("multi_mask")
    if rec.unknown_files:
        flags.append("unknown_files")
    if rec.id_mismatch:
        flags.append("id_mismatch")
    return "|".join(flags)


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_manifest(pairs: list[PatientRecord], folds: dict[str, int], root: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for rec in sorted(pairs, key=lambda r: r.pid):
            w.writerow({
                "pid": rec.pid,
                "fold": folds[rec.pid],
                "face_path": _rel(rec.pick("face_scan"), root),
                "mask_path": _rel(rec.pick("mask"), root),
                "template_path": _rel(rec.pick("template"), root),
                "datasheet_path": _rel(rec.pick("datasheet"), root),
                "has_template": int(rec.has_template),
                "has_metadata": int(rec.has_metadata),
                "source_category": rec.category,
                "n_face_files": rec.count("face_scan"),
                "n_mask_files": rec.count("mask"),
                "nested_in": rec.nested_in or "",
                "anomaly_flags": _anomaly_flags(rec),
            })


def write_simple_csv(records: list[PatientRecord], root: Path, out: Path, columns: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for rec in sorted(records, key=lambda r: r.pid):
            w.writerow({name: fn(rec, root) for name, fn in columns.items()})


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_index(
    root: Path,
    out_dir: Path,
    k: int = 5,
    seed: int = 42,
    expect: int | None = EXPECTED_COMPLETE_PAIRS,
    exclude_nested: bool = False,
) -> dict:
    """Index the archive, write all outputs, return the report dict.

    `exclude_nested` drops patient folders that sit inside another patient
    folder. Default is False — we would rather over-report and flag than
    silently discard a patient. See the note on 04358/04425 in _assert_expected.
    """
    records, warnings = discover_patients(root)
    if exclude_nested:
        dropped = [r.pid for r in records if r.nested_in]
        records = [r for r in records if not r.nested_in]
        if dropped:
            warnings.append(f"EXCLUDED_NESTED pids={dropped} (--exclude-nested was set)")
    pairs = [r for r in records if r.is_complete_pair]

    folds = assign_folds([r.pid for r in pairs], k=k, seed=seed)
    report = build_report(records, pairs, warnings, root, k, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(pairs, folds, root, out_dir / "manifest.csv")

    # RQ9 corpus: unpaired faces from the SAME scanner and population as the
    # training set — a domain-matched pretraining source, unlike a public
    # 3D-face dataset which would carry a domain gap to argue away.
    write_simple_csv(
        [r for r in records if r.has_face and not r.has_mask],
        root,
        out_dir / "pretrain_faces.csv",
        {
            "pid": lambda r, _: r.pid,
            "face_path": lambda r, rt: _rel(r.pick("face_scan"), rt),
            "source_category": lambda r, _: r.category,
        },
    )

    # Must-verify assumption #2: does a real template equal the negative of the
    # FINAL (carved) mask, or of the PRE-carving mask? These patients have a
    # template on file; those that also have a mask let us compare directly.
    write_simple_csv(
        [r for r in records if r.has_template],
        root,
        out_dir / "template_validation.csv",
        {
            "pid": lambda r, _: r.pid,
            "template_path": lambda r, rt: _rel(r.pick("template"), rt),
            "mask_path": lambda r, rt: _rel(r.pick("mask"), rt),
            "has_paired_mask": lambda r, _: int(r.has_mask),
            "source_category": lambda r, _: r.category,
        },
    )

    (out_dir / "index_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(report, out_dir)

    if expect is not None:
        _assert_expected(len(pairs), expect, report)

    return report


def _print_summary(report: dict, out_dir: Path) -> None:
    t = report["totals"]
    print("\n" + "=" * 68)
    print("DATASET INDEX SUMMARY")
    print("=" * 68)
    print(f"  archive root ............. {report['archive_root']}")
    print(f"  patient folders found .... {t['patient_folders_found']}")
    print(f"  COMPLETE PAIRS (face+mask) {t['complete_pairs_face_and_mask']}")
    print(f"    ...of which have template {t['pairs_with_template']}")
    print(f"    ...of which lack metadata {t['pairs_without_metadata']}")
    print(f"  face-only (RQ9 corpus) ... {t['face_only']}")
    print(f"  template but no mask ..... {t['template_but_no_mask']}")
    print("\n  complete pairs by source folder:")
    for cat, n in report["complete_pairs_by_category"].items():
        print(f"    {n:>5}  {cat}")

    anomalies = report["anomalies"]
    flagged = {key: val for key, val in anomalies.items() if val}
    if flagged:
        print("\n  ANOMALIES (recorded, not fatal — inspect index_report.json):")
        for key, val in flagged.items():
            print(f"    {key}: {len(val)}")
    else:
        print("\n  no anomalies detected.")
    print(f"\n  outputs written to: {out_dir}")
    print("  NOTE: these files contain patient IDs. Do not commit them.")
    print("=" * 68 + "\n")


def _assert_expected(found: int, expect: int, report: dict) -> None:
    """Halt if the pair count disagrees with the documented expectation.

    Rationale (from the project brief): if the count is not what the archive
    inventory says it should be, then either the pairing logic or the archive
    is wrong. Continuing would silently train on the wrong cohort, so we stop.
    """
    if found == expect:
        print(f"[OK] Complete-pair count matches expectation: {found} == {expect}\n")
        return

    print("=" * 68, file=sys.stderr)
    print(f"[HALT] Expected {expect} complete face+mask pairs, found {found}.", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    print("Diagnostic — complete pairs by source folder:", file=sys.stderr)
    for cat, n in report["complete_pairs_by_category"].items():
        print(f"    {n:>5}  {cat}", file=sys.stderr)
    print("\nMost likely causes, in order of likelihood:", file=sys.stderr)
    if found < expect:
        print(
            "  1. The walk was pointed at '1. Training/complete_file_set' only.\n"
            "     That folder holds 366. The remaining 2 (09751, 12441) live in\n"
            "     '1. Training/missing_datasheet' — they have face+mask but no\n"
            "     datasheet. Point --root at the ARCHIVE ROOT, not a subfolder.",
            file=sys.stderr,
        )
    else:
        print(
            "  1. A NESTED patient folder is being counted as its own patient.\n"
            "     The archive is known to contain folder 04425 inside folder\n"
            "     04358. If 04425 holds a genuine, distinct patient with both a\n"
            "     face and a mask, the true cohort is 369 and the long-standing\n"
            "     '368' is an undercount — a reportable finding, not a bug.\n"
            "     If it is a misfiled duplicate, re-run with --exclude-nested.\n"
            "     Check anomalies.nested_patient_folders in index_report.json and\n"
            "     inspect that folder before deciding. Do not guess: the cohort\n"
            "     size appears in the thesis.",
            file=sys.stderr,
        )
    print(
        "  2. The upload/unzip is incomplete — compare 'patient_folders_found'\n"
        "     in index_report.json against the ~1023 folders in the archive.",
        file=sys.stderr,
    )
    print(
        "  3. Filenames drifted from '<pid>_<kind>.<ext>'. Check\n"
        "     anomalies.unrecognised_filenames in index_report.json.",
        file=sys.stderr,
    )
    print(
        "\nIf the new count is genuinely correct, re-run with --expect <n> and\n"
        "record the change in the thesis methodology — the cohort size is a\n"
        "reported quantity, not an implementation detail.",
        file=sys.stderr,
    )
    sys.exit(2)


def extract_archive(zip_path: Path, dest: Path) -> Path:
    """Extract the Drive ZIP, returning the directory that holds the branches.

    Colab note: extract to the VM's local disk (/content/...), never to Drive.
    Thousands of small-file reads over the Drive FUSE mount are pathologically
    slow; local extraction turns a multi-hour walk into seconds.
    """
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    # If the ZIP wrapped everything in a single top folder, descend into it so
    # that --root always means "the folder containing '1. Training'".
    entries = [p for p in dest.iterdir() if not p.name.startswith("__MACOSX")]
    dirs = [p for p in entries if p.is_dir()]
    if len(dirs) == 1 and len(entries) == 1:
        print(f"[extract] descending into single top-level folder: {dirs[0].name}")
        return dirs[0]
    return dest


# --------------------------------------------------------------------------
# Self-test — validate the logic before touching real data
# --------------------------------------------------------------------------

def self_test() -> int:
    """Build a synthetic archive mirroring the real taxonomy and index it.

    This reproduces, in miniature, every structural feature that could break
    the indexer: the 366+2 split across two folders, the nested-patient
    anomaly, the excluded Unterkiefer branch, unpaired faces, and a
    mask-without-face. No patient data is involved.
    """
    print("Running self-test on a synthetic archive (no real data used)...\n")
    tmp = Path(tempfile.mkdtemp(prefix="index_selftest_"))
    try:
        root = tmp / "KI_Assistenz_File_Repository"

        def make(folder: str, pid: str, kinds: list[str]) -> None:
            d = root / folder / pid
            d.mkdir(parents=True, exist_ok=True)
            for kind in kinds:
                ext = "png" if kind == "datasheet" else "stl"
                (d / f"{pid}_{kind}.{ext}").write_text("synthetic")

        full = ["face_scan", "template", "mask", "datasheet"]

        # 366 complete patients at the top level of complete_file_set.
        for i in range(366):
            make("1. Training/complete_file_set", f"{10000 + i:05d}", full)

        # The nesting anomaly, mirroring 04425-inside-04358. Modelled here as
        # an INCOMPLETE nested folder (face only), which is the configuration
        # consistent with the archive's documented total of 368. If the real
        # 04425 turns out to hold a complete pair, the indexer will report 369
        # and halt with a diagnostic — by design.
        nested_child = root / "1. Training/complete_file_set" / "10000" / "10999"
        nested_child.mkdir(parents=True, exist_ok=True)
        (nested_child / "10999_face_scan.stl").write_text("synthetic")

        # The 2 patients with face+mask but NO datasheet — the crux of 366+2.
        make("1. Training/missing_datasheet", "09751", ["face_scan", "template", "mask"])
        make("1. Training/missing_datasheet", "12441", ["face_scan", "template", "mask"])

        # Non-pairs.
        for i in range(28):
            make("2. Testing/face_template_only", f"{20000 + i:05d}", ["face_scan", "template", "datasheet"])
        for i in range(584):
            make("3. Outliers/only_face", f"{30000 + i:05d}", ["face_scan"])
        for i in range(3):
            make("3. Outliers/only_mask", f"{40000 + i:05d}", ["mask"])

        # Excluded branch — must NOT be indexed even though it looks valid.
        make("4. Unterkiefer Files", "99999", ["face_scan", "mask"])

        out_dir = tmp / "out"
        report = run_index(root, out_dir, k=5, seed=42, expect=None)

        t = report["totals"]
        checks = [
            ("complete pairs == 368", t["complete_pairs_face_and_mask"], 368),
            ("face-only == 613", t["face_only"], 28 + 584 + 1),
            ("mask-only == 3", t["mask_only"], 3),
            ("pairs lacking metadata == 2", t["pairs_without_metadata"], 2),
            ("nested anomaly detected == 1", len(report["anomalies"]["nested_patient_folders"]), 1),
            ("Unterkiefer excluded", 99999 in [int(p) for p in _manifest_pids(out_dir)], False),
        ]

        # Folds must be balanced and cover every pair exactly once.
        folds = _manifest_folds(out_dir)
        sizes = Counter(folds.values())
        checks.append(("all 368 pairs assigned a fold", len(folds), 368))
        checks.append(("fold sizes balanced within 1", max(sizes.values()) - min(sizes.values()) <= 1, True))

        # Fold assignment must be reproducible.
        again = assign_folds(list(folds), k=5, seed=42)
        checks.append(("fold assignment reproducible", again == folds, True))
        different = assign_folds(list(folds), k=5, seed=7)
        checks.append(("different seed -> different folds", different != folds, True))

        print("Self-test assertions:")
        failures = 0
        for label, got, want in checks:
            ok = got == want
            failures += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}  (got {got!r}, want {want!r})")

        print("\nSELF-TEST " + ("PASSED" if not failures else f"FAILED ({failures} assertion(s))"))
        return 0 if not failures else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _manifest_pids(out_dir: Path) -> list[str]:
    with (out_dir / "manifest.csv").open(encoding="utf-8") as fh:
        return [row["pid"] for row in csv.DictReader(fh)]


def _manifest_folds(out_dir: Path) -> dict[str, int]:
    with (out_dir / "manifest.csv").open(encoding="utf-8") as fh:
        return {row["pid"]: int(row["fold"]) for row in csv.DictReader(fh)}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Index the KI-Assistenz archive and emit the CV manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--root", type=Path, help="Archive root (the folder containing '1. Training').")
    src.add_argument("--zip", type=Path, help="Drive ZIP to extract first.")
    ap.add_argument("--extract-to", type=Path, default=Path("/content/KI_Assistenz_File_Repository"),
                    help="Where to extract --zip (use local VM disk, not Drive).")
    ap.add_argument("--out-dir", type=Path, default=Path("./index_out"),
                    help="Where to write manifest.csv and friends.")
    ap.add_argument("-k", "--folds", type=int, default=5, help="Number of CV folds (default 5).")
    ap.add_argument("--seed", type=int, default=42, help="Fold-assignment seed (default 42).")
    ap.add_argument("--expect", type=int, default=EXPECTED_COMPLETE_PAIRS,
                    help=f"Expected complete-pair count (default {EXPECTED_COMPLETE_PAIRS}).")
    ap.add_argument("--no-assert", action="store_true",
                    help="Report the count but do not halt on mismatch.")
    ap.add_argument("--exclude-nested", action="store_true",
                    help="Drop patient folders nested inside another patient folder "
                         "(see the 04358/04425 anomaly). Off by default.")
    ap.add_argument("--self-test", action="store_true",
                    help="Validate the indexer on a synthetic archive and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.zip:
        root = extract_archive(args.zip, args.extract_to)
    elif args.root:
        root = args.root
    else:
        ap.error("one of --root, --zip or --self-test is required")

    run_index(
        root=root,
        out_dir=args.out_dir,
        k=args.folds,
        seed=args.seed,
        expect=None if args.no_assert else args.expect,
        exclude_nested=args.exclude_nested,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
