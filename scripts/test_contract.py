"""Test transformer baseline contract manually."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import validate_evaluation_contract


def test_transformer_baseline_contract(output_root: Path) -> None:
    """Validate transformer baseline produces required artifacts."""
    print(f"Testing transformer contract with output_root: {output_root}")

    # Validate phase-5 compatible evaluation artifacts
    try:
        validate_evaluation_contract(output_root)
        print("✓ Phase-5 compatible evaluation artifacts validated")
    except Exception as e:
        print(f"✗ Evaluation contract validation failed: {e}")
        return False

    # Validate transformer-specific model artifacts
    model_dir = output_root / "model"
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False
    print(f"✓ Model directory exists: {model_dir}")

    required_model_files = [
        "config.json",
        "model.safetensors",  # or pytorch_model.bin for older versions
        "tokenizer_config.json",
        "tokenizer.json",  # or vocab.txt
    ]

    for filename in required_model_files:
        model_file = model_dir / filename
        if not model_file.exists():
            print(f"✗ Required model file not found: {filename}")
            return False
        print(f"✓ Model file exists: {filename}")

    # Validate model config
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "num_labels" not in config:
        print("✗ Model config missing num_labels")
        return False
    if config["num_labels"] != 2:
        print(f"✗ Expected num_labels=2, got {config['num_labels']}")
        return False
    print(f"✓ Model config valid (num_labels={config['num_labels']})")

    # Validate run_metadata has transformer-specific fields
    metadata_path = output_root / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["model_family"] != "transformer":
        print(f"✗ Expected model_family=transformer, got {metadata['model_family']}")
        return False
    print("✓ Model family is transformer")

    if "model_name" not in metadata:
        print("✗ Model name not in metadata")
        return False
    print(f"✓ Model name: {metadata['model_name']}")

    if "model_config" not in metadata:
        print("✗ Model config not in metadata")
        return False
    print("✓ Model config present")

    if "device" not in metadata["model_config"]:
        print("✗ Device not in model config")
        return False
    print(f"✓ Device: {metadata['model_config']['device']}")

    print("\n✓ All transformer contract tests passed!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", required=True, help="Path to transformer output artifacts"
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    success = test_transformer_baseline_contract(output_root)
    sys.exit(0 if success else 1)
