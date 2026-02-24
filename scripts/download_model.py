"""
Script to securely download the model weights from Weights & Biases Artifacts.
Designed to be executed during the CI/CD Docker multi-stage build process.
"""
import os
import sys
from pathlib import Path

import wandb
from wandb.errors import CommError

# Assuming we run this from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ARTIFACT_PATH = "your_wandb_workspace/your_project/neuroseg_model:latest"

def fetch_model_from_registry(artifact_path: str, output_dir: Path) -> None:
    """
    Connects to W&B, authenticates using WANDB_API_KEY env var, and downloads the model checkpoint.
    
    Args:
        artifact_path: W&B path to the artifact (e.g., entity/project/name:version)
        output_dir: Local directory to save the downloaded weights.
    """
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("CRITICAL: WANDB_API_KEY environment variable is not set.", file=sys.stderr)
        print("Model download aborted.", file=sys.stderr)
        sys.exit(1)

    print(f"INFO: Attempting to download artifact: {artifact_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize W&B API (automatically picks up WANDB_API_KEY)
        api = wandb.Api()
        artifact = api.artifact(artifact_path, type="model")
        
        print(f"INFO: Artifact found. Size: {artifact.size / (1024*1024):.2f} MB")
        
        # Download the specific .ckpt file (assuming it's named last.ckpt or similar inside the 
        # artifact)
        # We use download() which pulls the whole directory/files attached to that version
        download_path = artifact.download(root=str(output_dir))
        
        print(f"SUCCESS: Model downloaded successfully to: {download_path}")

    except CommError as ce:
        print(f"ERROR: Network or Authentication failure with W&B API: {ce}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED ERROR: Failed to download model from registry: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # You can override the artifact path via environment variables for different environments 
    # (staging/prod)
    target_artifact = os.getenv("WANDB_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH)
    
    print("--- Starting Model Registry Fetch Process ---")
    fetch_model_from_registry(artifact_path=target_artifact, output_dir=MODELS_DIR)
    print("--- Fetch Process Complete ---")