import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Wav2Vec CTC model.")
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to the training data directory.",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        required=True,
        help="Path to the validation data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for fine-tuning.",
    )
    return parser.parse_args()