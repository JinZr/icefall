import argparse
import json
import os
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def text_normalize(str_line: str):
    line = str_line.strip().rstrip("\n")
    line = line.replace(" ", "")
    line = line.replace("<sil>", "")
    line = line.replace("<%>", "")
    line = line.replace("<->", "")
    line = line.replace("<$>", "")
    line = line.replace("<#>", "")
    line = line.replace("<_>", "")
    line = line.replace("<space>", "")
    line = line.replace("`", "")
    line = line.replace("&", "")
    line = line.replace(",", "")
    line = line.replace("Ａ", "")
    line = line.replace("ａ", "A")
    line = line.replace("ｂ", "B")
    line = line.replace("ｃ", "C")
    line = line.replace("ｋ", "K")
    line = line.replace("ｔ", "T")
    line = line.replace("，", "")
    line = line.replace("丶", "")
    line = line.replace("。", "")
    line = line.replace("、", "")
    line = line.replace("？", "")
    line = line.replace("·", "")
    line = line.replace("*", "")
    line = line.replace("！", "")
    line = line.replace("$", "")
    line = line.replace("+", "")
    line = line.replace("-", "")
    line = line.replace("\\", "")
    line = line.replace("?", "")
    line = line.replace("￥", "")
    line = line.replace("%", "")
    line = line.replace(".", "")
    line = line.replace("<", "")
    line = line.replace("＆", "")
    line = line.upper()

    return line


def main():
    parser = argparse.ArgumentParser(
        description="Split audio by JSON utterance boundaries, resample to 16kHz, and prepare Kaldi data files"
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing JSON files with utterances",
    )
    parser.add_argument(
        "--audio-dir", required=True, help="Directory containing WAV audio files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory for split WAVs and Kaldi files",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    audio_dir = Path(args.audio_dir)
    root_out = Path(args.output_dir)

    for json_path in tqdm(json_dir.glob("*.json")):
        if json_path.stem.startswith("."):
            continue
        # Directory for Kaldi label files
        kaldi_dir = root_out / "kaldi"
        kaldi_dir.mkdir(parents=True, exist_ok=True)

        patient_id = json_path.stem
        audio_path = audio_dir / f"{patient_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and resample audio to 16kHz
        audio, sr = librosa.load(str(audio_path), sr=16000)

        # Read utterances
        try:
            with open(json_path, "r", encoding="utf-8") as j:
                data = json.load(j)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error reading JSON file {json_path}: {e}")

        # Create per‑speaker folders for WAV segments
        speaker_key = str(data.get("speaker_id", json_path.stem))
        patient_wav_dir = root_out / "wav" / speaker_key
        investigator_wav_dir = root_out / "wav" / "investigator" / speaker_key
        patient_wav_dir.mkdir(parents=True, exist_ok=True)
        investigator_wav_dir.mkdir(parents=True, exist_ok=True)

        utterances = data.get("utterances", [])
        # utterances = sorted(utterances, key=lambda x: x.get("start_time", 0.0))

        # Prepare Kaldi entries
        wav_scp = []
        text = []
        text_normalized = []
        utt2spk = []

        for idx, utt in enumerate(utterances, start=1):
            spkr = utt.get("speaker_id", "Other")
            if spkr == "Speaker 1":
                spk_label = f"{speaker_key}_INV"
            elif spkr == "Speaker 2":
                spk_label = f"{speaker_key}_PAR"
            else:
                spk_label = f"{speaker_key}_{spkr.replace(' ', '_').upper()}"

            start_sec = utt.get("start_time", 0.0)
            end_sec = utt.get("end_time", start_sec)
            start_sample = int(start_sec * sr)
            end_sample = int((end_sec + 0.1) * sr)
            segment = audio[start_sample:end_sample]

            utt_id = f"{patient_id}_{spk_label}_{idx:04d}"
            utt_id = utt_id.replace(" ", "_").upper()

            out_dir = patient_wav_dir if spkr == "Speaker 2" else investigator_wav_dir
            out_wav = out_dir / f"{utt_id}.wav"
            sf.write(str(out_wav), segment, sr)

            # Record entries
            wav_scp.append(f"{utt_id} {out_wav.resolve()}")
            transcript = utt.get("transcript", "").strip()
            text.append(f"{utt_id} {transcript}")
            text_normalized.append(f"{utt_id} {text_normalize(transcript)}")
            utt2spk.append(f"{utt_id} {spk_label}")

        # Write Kaldi data files in separate directory
        with open(kaldi_dir / "wav.scp", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(wav_scp, key=lambda x: x.split(" ")[0])) + "\n")
        with open(kaldi_dir / "text_orig", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(text, key=lambda x: x.split(" ")[0])) + "\n")
        with open(kaldi_dir / "text", "a", encoding="utf-8") as f:
            f.write(
                "\n".join(sorted(text_normalized, key=lambda x: x.split(" ")[0])) + "\n"
            )
        with open(kaldi_dir / "utt2spk", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(utt2spk, key=lambda x: x.split(" ")[0])) + "\n")


if __name__ == "__main__":
    main()
