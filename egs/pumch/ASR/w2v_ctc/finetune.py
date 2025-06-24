#!/usr/bin/env python
# fine_tune_ssl.py
# Fine‑tune wav2vec‑family (or any CTC SSL) on a Hugging Face CSV corpus.
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import evaluate
import numpy as np
import torch
from collator import DataCollatorCTCWithPadding
from datasets import Audio, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers.utils import logging

logging.set_verbosity_info()


# ──────────────────── CLI ────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-csv", required=True, type=str, help="Path to the training CSV file"
    )
    p.add_argument(
        "--valid-csv", required=True, type=str, help="Path to the validation CSV file"
    )
    p.add_argument("--model-name", default="facebook/wav2vec2-base")
    p.add_argument("--output-dir", default="./w2v_ctc/exp", type=Path)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--push-to-hub", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    # ─────────── Load CSV and describe its columns ───────────
    train_ds = load_dataset(
        "csv",
        data_files=args.train_csv,
        column_names=[
            "id",
            "speaker",
            "path",
            "text",
        ],  # 'path' → audio file, 'text' → transcript
        split="train",
    )
    valid_ds = load_dataset(
        "csv",
        data_files=args.valid_csv,
        column_names=[
            "id",
            "speaker",
            "path",
            "text",
        ],  # 'path' → audio file, 'text' → transcript
        split="train",
    )

    # Cast the 'path' column so each example lazily loads the waveform.
    train_ds = train_ds.cast_column("path", Audio(sampling_rate=16_000)).remove_columns(
        ["speaker", "id"]
    )
    valid_ds = valid_ds.cast_column("path", Audio(sampling_rate=16_000)).remove_columns(
        ["speaker", "id"]
    )

    train_ds = train_ds.rename_column("path", "input_values")
    valid_ds = valid_ds.rename_column("path", "input_values")

    train_ds = train_ds.rename_column("text", "labels")
    valid_ds = valid_ds.rename_column("text", "labels")

    tokenizer = Wav2Vec2CTCTokenizer(
        "./data/tokens.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    processor = Wav2Vec2Processor.from_pretrained(
        args.model_name,
        tokenizer=tokenizer,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(
            tokenizer
        ),  # Ensure the model's vocab size matches the tokenizer
    )
    model.freeze_feature_extractor()  # Freeze the feature extractor

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_total_limit=3,
        logging_steps=50,
        push_to_hub=args.push_to_hub,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    processor.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)  # includes config & weights

    print("✔ Done — fine-tuned model saved to", args.output_dir)


if __name__ == "__main__":
    main()
