import argparse
import json
import os
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


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
        # Directories for WAV segments and Kaldi labels
        wav_dir = root_out / "wav"
        kaldi_dir = root_out / "kaldi"
        wav_dir.mkdir(parents=True, exist_ok=True)
        kaldi_dir.mkdir(parents=True, exist_ok=True)

        patient_id = json_path.stem
        audio_path = audio_dir / f"{patient_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load and resample audio to 16kHz
        audio, sr = librosa.load(str(audio_path), sr=16000)

        # Read utterances
        with open(json_path, "r", encoding="utf-8") as j:
            data = json.load(j)
        utterances = data.get("utterances", [])
        # utterances = sorted(utterances, key=lambda x: x.get("start_time", 0.0))

        # Prepare Kaldi entries
        wav_scp = []
        text = []
        utt2spk = []

        for idx, utt in enumerate(utterances, start=1):
            spkr = utt.get("speaker_id", "Other")
            if spkr == "Speaker 1":
                spk_label = "INV"
            elif spkr == "Speaker 2":
                spk_label = "PAR"
            else:
                spk_label = spkr.replace(" ", "_").upper()

            start_sec = utt.get("start_time", 0.0)
            end_sec = utt.get("end_time", start_sec)
            start_sample = int(start_sec * sr)
            end_sample = int((end_sec + 0.1) * sr)
            segment = audio[start_sample:end_sample]

            utt_id = f"{patient_id}_{spk_label}_{idx:04d}"
            out_wav = wav_dir / f"{utt_id}.wav"
            sf.write(str(out_wav), segment, sr)

            # Record entries
            wav_scp.append(f"{utt_id} {out_wav.resolve()}")
            transcript = utt.get("transcript", "").strip()
            text.append(f"{utt_id} {transcript}")
            utt2spk.append(f"{utt_id} {spk_label}")

        # Write Kaldi data files in separate directory
        with open(kaldi_dir / "wav.scp", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(wav_scp, key=lambda x: x.split(" ")[0])) + "\n")
        with open(kaldi_dir / "text", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(text, key=lambda x: x.split(" ")[0])) + "\n")
        with open(kaldi_dir / "utt2spk", "a", encoding="utf-8") as f:
            f.write("\n".join(sorted(utt2spk, key=lambda x: x.split(" ")[0])) + "\n")


if __name__ == "__main__":
    main()
