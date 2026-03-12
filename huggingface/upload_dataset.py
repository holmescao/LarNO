"""
Upload LarNO benchmark dataset to HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env variable

Usage:
    python upload_dataset.py

Benchmark layout (relative to this script):
    ../benchmark/urbanflood/
        geodata/
            region1_20m/
            ukea_8m_5min/
            ukea_2m_5min/
        flood/
            region1_20m/     (event1 ~ event80, 80 events, ~large)
            ukea_8m_5min/    (20 events, training resolution)
            ukea_2m_5min/    (20 events, zero-shot super-res test)
"""

from huggingface_hub import HfApi, create_repo
import os

# ── Config ────────────────────────────────────────────────────────────────────
HF_USERNAME    = "holmescao"
DATASET_REPO   = f"{HF_USERNAME}/LarNO-dataset"
BENCHMARK_ROOT = "../benchmark/urbanflood"   # relative to this script
DATASET_CARD   = "dataset_card.md"

# Which sub-datasets to upload
UPLOAD_GEODATA       = True
UPLOAD_UKEA_8M       = True   # training resolution (20 events)
UPLOAD_UKEA_2M       = True   # zero-shot super-res test (20 events)
UPLOAD_REGION1_20M   = True   # Futian large dataset (80 events)

# ── Create repo ───────────────────────────────────────────────────────────────
api = HfApi()

print(f"Creating dataset repo: {DATASET_REPO}")
create_repo(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    exist_ok=True,
    private=False,
)

# ── Upload Dataset Card as README.md ──────────────────────────────────────────
if os.path.exists(DATASET_CARD):
    print("Uploading Dataset Card ...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD,
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Add Dataset Card",
    )

def upload_folder(local_path, repo_path, msg):
    if os.path.exists(local_path):
        print(f"Uploading {repo_path} ...")
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=repo_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=msg,
        )
        print(f"  -> {repo_path} done.")
    else:
        print(f"WARNING: not found: {local_path}")

# ── Upload geodata ────────────────────────────────────────────────────────────
if UPLOAD_GEODATA:
    upload_folder(
        os.path.join(BENCHMARK_ROOT, "geodata"),
        "geodata",
        "Add geodata (DEM, boundary) for all datasets",
    )

# ── Upload UKEA 8m (training resolution) ─────────────────────────────────────
if UPLOAD_UKEA_8M:
    upload_folder(
        os.path.join(BENCHMARK_ROOT, "flood", "ukea_8m_5min"),
        "flood/ukea_8m_5min",
        "Add UKEA 8m/5min dataset (20 events, training resolution)",
    )

# ── Upload UKEA 2m (zero-shot super-res test) ─────────────────────────────────
if UPLOAD_UKEA_2M:
    upload_folder(
        os.path.join(BENCHMARK_ROOT, "flood", "ukea_2m_5min"),
        "flood/ukea_2m_5min",
        "Add UKEA 2m/5min dataset (20 events, zero-shot super-res test)",
    )

# ── Upload Futian region1_20m ─────────────────────────────────────────────────
if UPLOAD_REGION1_20M:
    upload_folder(
        os.path.join(BENCHMARK_ROOT, "flood", "region1_20m"),
        "flood/region1_20m",
        "Add Futian region1 20m/5min dataset (80 events)",
    )

print("\nDone! Dataset repo:", f"https://huggingface.co/datasets/{DATASET_REPO}")
