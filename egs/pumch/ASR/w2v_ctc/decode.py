#!/usr/bin/env python
# decode_and_eval.py
# Decode a CTC model fine-tuned by finetune.py and compute WER.

import argparse
import json
import re
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from collator import DataCollatorCTCWithPadding  # 与 finetune.py 相同的打包器
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# 与训练脚本保持一致的中日韩字符逐字分词
_CJK_RE = re.compile(
    r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF"
    r"\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
)


def tokenize_by_CJK_char(line: str) -> str:
    chars = _CJK_RE.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test-csv",
        required=True,
        type=str,
        help="待评估的 CSV (id speaker path text)",
    )
    p.add_argument(
        "--exp-dir",
        required=True,
        type=Path,
        help="finetune.py 训练产出的目录（含 config/模型权重/tokenizer)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        choices=["greedy_search", "beam_search"],
    )
    return p.parse_args()


def main():
    args = get_args()
    output_dir = args.exp_dir / args.decoding_method
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据集加载 -----------------------------------------------------------------
    ds = load_dataset(
        "csv",
        data_files=args.test_csv,
        column_names=["id", "speaker", "path", "text"],
        split="train",
    )
    ds = (
        ds.cast_column("path", Audio(sampling_rate=16_000))
        .remove_columns(["speaker"])
        .rename_column("path", "input_values")
    )

    # 2. 模型与处理器 ---------------------------------------------------------------
    processor = Wav2Vec2Processor.from_pretrained(args.exp_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.exp_dir).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 3. 推断 -----------------------------------------------------------------------
    utt_ids, utt_paths, predictions, references = [], [], [], []
    for batch in tqdm(ds.iter(batch_size=args.batch_size), desc="Decoding"):
        for input_value in batch["input_values"]: 
            utt_paths.append(input_value["path"])
            inputs = processor(
                input_value["array"],
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                logits = model(
                    inputs.input_values.to(model.device),
                ).logits
            ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(ids)
            predictions.extend(preds)
        utt_ids.extend(batch["id"])
        references.extend(batch["text"])

    # 4. 计算 WER -------------------------------------------------------------------
    refs_tokenized = list(map(tokenize_by_CJK_char, references))
    hyps_tokenized = list(map(tokenize_by_CJK_char, predictions))
    wer_metric = evaluate.load("wer")
    wer = float(
        wer_metric.compute(predictions=hyps_tokenized, references=refs_tokenized)
    )

    # 5. 保存结果 -------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "ids": utt_ids,
            "reference": references,
            "hypothesis": predictions,
            "paths": utt_paths,
        }
    )
    df.to_csv(output_dir / "decoded.csv", index=False, encoding="utf-8")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"wer": wer}, fp, indent=2, ensure_ascii=False)

    print(f"✔ Decoding finished — WER = {wer:.4%}")
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
