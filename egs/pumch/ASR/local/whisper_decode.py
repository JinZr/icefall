import argparse
import os
from pathlib import Path

import whisper
from jiwer import cer, wer
from tqdm import tqdm


def load_kaldi_data(data_dir):
    """
    Load Kaldi-formatted dataset. Expects wav.scp and text files in data_dir.
    Returns:
      utt2wav: dict of utt_id -> wav_path
      refs: dict of utt_id -> reference transcript
    """
    wav_scp = Path(data_dir) / "wav.scp"
    text_file = Path(data_dir) / "text"

    if not wav_scp.exists() or not text_file.exists():
        raise FileNotFoundError("wav.scp or text file not found in data_dir")

    utt2wav = {}
    with open(wav_scp, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, wav_path = parts
            utt2wav[utt_id] = wav_path

    refs = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, transcript = parts
            refs[utt_id] = transcript

    # intersect utt IDs in both
    common = set(utt2wav.keys()) & set(refs.keys())
    utt2wav = {u: utt2wav[u] for u in common}
    refs = {u: refs[u] for u in common}
    return utt2wav, refs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode a Kaldi-formatted dataset with Whisper and compute WER/CER, saving results."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to Kaldi data directory containing wav.scp and text",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        help="Whisper model size (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run Whisper on"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code (e.g., en) to pass to Whisper",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="decode_results.tsv",
        help="TSV file to save utt_id, ref, hyp, WER, CER",
    )
    args = parser.parse_args()

    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)

    utt2wav, refs = load_kaldi_data(args.data_dir)
    print(f"Found {len(utt2wav)} utterances.")

    # Ensure output directory exists
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_wer = 0.0
    total_cer = 0.0
    count = 0

    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write("utt_id\tref\thyp\twer\tcer\n")

        for utt_id, wav_path in tqdm(utt2wav.items()):
            if not os.path.exists(wav_path):
                print(f"Warning: wav file {wav_path} not found, skipping.")
                continue
            result = model.transcribe(
                wav_path, language=args.language, initial_prompt="以下是普通话的句子"
            )
            hyp = result["text"].strip()
            ref = refs[utt_id].strip()

            utt_wer = wer(ref, hyp)
            utt_cer = cer(ref, hyp)
            total_wer += utt_wer
            total_cer += utt_cer
            count += 1

            print(f"{utt_id}: WER={utt_wer:.3f}, CER={utt_cer:.3f}")
            # write to TSV
            print(f"{utt_id}\t{ref}\t{hyp}\t{utt_wer:.3f}\t{utt_cer:.3f}\n")
            outf.write(f"{utt_id}\t{ref}\t{hyp}\t{utt_wer:.3f}\t{utt_cer:.3f}\n")

        if count > 0:
            avg_wer = total_wer / count
            avg_cer = total_cer / count
            print(f"\nDecoded {count} utterances.")
            print(f"Average WER: {avg_wer:.3f}")
            print(f"Average CER: {avg_cer:.3f}")
            print(f"Results saved to {args.output_file}")
        else:
            print("No utterances decoded.")
