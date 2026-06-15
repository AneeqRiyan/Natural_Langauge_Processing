#!/usr/bin/env python3
"""
Entry point for SemEval 2026 Task 5 (AmbiStory) evaluation.

Usage:
    python predict.py <input_json> <output_jsonl>

Arguments:
    input_json:  Path to test data JSON file (same format as train/dev data)
    output_jsonl: Path to write predictions (JSONL format: one {"id": "...", "prediction": N} per line)
"""

import sys
import json
import torch
import yaml
from pathlib import Path

from src.inference import InferenceModel


def main():
    """Main entry point for predictions."""
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_json> <output_jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load input data
    with open(input_path) as f:
        data = json.load(f)

    # Load model checkpoint
    checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please run training first: python src/train.py config.yaml")
        sys.exit(1)

    # Initialize inference model
    inference_model = InferenceModel(str(checkpoint_path), config)

    # Generate predictions
    with open(output_path, "w") as f_out:
        for sample_id, sample in data.items():
            # Extract fields from sample
            precontext = sample.get("precontext", "")
            sentence = sample.get("sentence", "")
            meaning = sample.get("judged_meaning", "")
            ending = sample.get("ending", "")

            # Get prediction
            result = inference_model.predict_single(
                precontext=precontext,
                sentence=sentence,
                meaning=meaning,
                ending=ending,
            )

            prediction = result["predicted_score"]
            # Round to nearest integer in range [1, 5]
            prediction = int(round(max(1, min(5, prediction))))

            # Write output in JSONL format
            output_line = json.dumps({"id": str(sample_id), "prediction": prediction})
            f_out.write(output_line + "\n")

    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
