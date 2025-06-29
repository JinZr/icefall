#!/usr/bin/env python
# fine_tune_ssl.py
# Fine‑tune wav2vec‑family (or any CTC SSL) on a Hugging Face CSV corpus.
import argparse
import re
from pathlib import Path

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


def tokenize_by_CJK_char(line: str) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return characters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


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
    p.add_argument("--exp-dir", default="./w2v_ctc/exp", type=Path)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-5)
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
        keep_in_memory=False,
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
        keep_in_memory=False,
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
        vocab_size=len(tokenizer),
    )
    model.freeze_feature_extractor()  # Freeze the feature extractor

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        pred_str = list(map(tokenize_by_CJK_char, pred_str))
        label_str = list(map(tokenize_by_CJK_char, label_str))
        print(pred_str)
        print(label_str)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_total_limit=3,
        logging_steps=50,
        push_to_hub=args.push_to_hub,
        report_to=["tensorboard"],
        warmup_steps=1000,
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
