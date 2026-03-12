"""
Upload LarNO pre-trained weights to HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env variable

Usage:
    python upload_weights.py
"""

from huggingface_hub import HfApi, create_repo
import os
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
HF_USERNAME   = "holmescao"          # your HuggingFace username
MODEL_REPO    = f"{HF_USERNAME}/LarNO"
WEIGHTS_ZIP   = "../exp_weights.zip"      # path to the weights zip
MODEL_CARD    = "model_card.md"           # Model Card (this directory)

# ── Create repo ───────────────────────────────────────────────────────────────
api = HfApi()

print(f"Creating model repo: {MODEL_REPO}")
create_repo(
    repo_id=MODEL_REPO,
    repo_type="model",
    exist_ok=True,
    private=False,
)

# ── Upload Model Card as README.md ────────────────────────────────────────────
print("Uploading Model Card ...")
api.upload_file(
    path_or_fileobj=MODEL_CARD,
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    repo_type="model",
    commit_message="Add Model Card",
)

# ── Upload weights zip ────────────────────────────────────────────────────────
if os.path.exists(WEIGHTS_ZIP):
    print(f"Uploading weights: {WEIGHTS_ZIP} ...")
    api.upload_file(
        path_or_fileobj=WEIGHTS_ZIP,
        path_in_repo="exp_weights.zip",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Add pre-trained Futian (region1_20m) checkpoint",
    )
    print("Done! Model repo:", f"https://huggingface.co/{MODEL_REPO}")
else:
    print(f"WARNING: weights zip not found at {WEIGHTS_ZIP}")
    print("Please update WEIGHTS_ZIP path and re-run.")
