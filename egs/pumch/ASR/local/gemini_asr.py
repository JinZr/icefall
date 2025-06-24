#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
from pathlib import Path

from google import genai
from tqdm import tqdm

# A minimal prompt for English transcription
# EN_PROMPT = (
#     "Please transcribe the following English audio verbatim. "
#     "Output only the transcript, without any extra metadata."
# )
EN_PROMPT = """You are given 72 canonical reference sentences that children with cleft-palate read for articulation assessment.

When you receive an inaccurately-pronounced utterance, output the single sentence from the list below that best matches what the speaker intended.  
Return that sentence verbatim identical wording and punctuation, nothing else.

Reference sentences (unordered):

Sixty six.
Aaron lives in an igloo.
Baby bell, baby bell.
Big red truck.
Billy Bob will play ball.
Bob is a baby boy.
Bobby and Bill play ball.
Buy baby a bib.
Charlie chew gum.
Chase the chickens.
Children watch a soccer match.
Choo choo ch.
Choo choo train.
Dig to Daddy.
Do it for daddy.
Get Kate a cake and a cookie.
Give Kate the cookie.
Go get the wagon.
Hannah hurt her hand.
I can laugh.
I can sing a song.
I do, you do, we do.
I have a firefly.
I have five fingers.
I like cake.
I like cheese pizza.
I like ice cream.
I like pizza.
I love caramel.
I see the sky.
I see the star.
I see the sun.
I'm going away.
Johnny told a joke.
Jim and Charlie chew gum.
Jimmy and Charlie chew gum.
Mama makes muffins.
Mary knew no one.
No one is coming.
Papa plays baseball.
Put the baby in the buggy.
Roll the carpet.
See the busy bees.
She goes to the shop.
She went shopping.
shine the shoes.
Sissy sees the stars.
Something smells funny.
Take Teddy to town.
Teach me to sing.
The feather fell off the leaf.
The puppy plays with a rope.
The puppy plays with the rope.
The puppy will pull a rope.
The zebra lives at the zoo.
The zebra lives in the zoo.
This is Tuesday.
Today is a good day.
Very fresh fruit.
Very funny.
Vicky drives a bus.
Vicky drives a van.
Wash the shoes.
Where is the way home?
You ran a long mile.
feed the fish.
fifty fifty
hurt her hand
say sixty six
shine the shoes.
sixty six
Chase the chickens."""



def get_args():
    p = argparse.ArgumentParser(
        description="Transcribe all English WAVs in a folder via Gemini "
        "and emit Kaldi-format text, wav.scp, utt2spk."
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing .wav files to transcribe",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where 'text', 'wav.scp', and 'utt2spk' will be written",
    )
    p.add_argument(
        "--sk-token",
        type=Path,
        default=Path("./local/sk_token"),
        help="Path to file containing your Gemini API key",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of concurrent Gemini requests to run (batch size)",
    )
    return p.parse_args()


def main():
    args = get_args()
    audio_dir = args.audio_dir
    out_dir = args.output_dir
    sk_path = args.sk_token

    if not audio_dir.is_dir():
        raise SystemExit(
            f"Error: audio-dir {audio_dir} does not exist or is not a directory."
        )

    wavs = sorted(audio_dir.glob("*/*.wav"))
    filtered_wavs = []
    for wav in wavs:
        if wav.stat().st_size > 0:
            filtered_wavs.append(wav)
    wavs = filtered_wavs
    if not wavs:
        raise SystemExit(f"No .wav files found in {audio_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # preload already processed utterance IDs
    processed = set()
    wavscp_path = out_dir / "wav.scp"
    if wavscp_path.exists():
        with open(wavscp_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.strip().split()
                if parts:
                    processed.add(parts[0])
    text_f = open(
        out_dir / "text", "w" if (out_dir / "text").exists() else "w", encoding="utf-8"
    )
    wavscp_f = open(wavscp_path, "w" if wavscp_path.exists() else "w", encoding="utf-8")
    utt2spk_f = open(
        out_dir / "utt2spk",
        "w" if (out_dir / "utt2spk").exists() else "w",
        encoding="utf-8",
    )

    # load API key
    with open(sk_path, "r") as f:
        sk_token = f.read().strip()

    def transcribe_wav(wav_path: Path):
        """
        Upload one WAV and get its transcript.
        Returns (utt_id, transcript, wav_path) or None on failure.
        """
        utt_id = wav_path.stem
        if utt_id.startswith("."):
            return None  # skip hidden files and dot‑files

        # Create a fresh Gemini client inside this worker
        worker_client = genai.Client(api_key=sk_token)

        try:
            upload = worker_client.files.upload(file=str(wav_path))
            resp = worker_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[EN_PROMPT, upload],
            )
        except Exception as e:
            tqdm.write(f"[!] Failed {utt_id}: {e}")
            return None
        try:
            return utt_id, resp.text.strip(), wav_path
        except AttributeError:
            tqdm.write(f"[!] Failed {utt_id}: no text in response")
            return utt_id, "", wav_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(transcribe_wav, wav): wav
            for wav in wavs
            if wav.stem not in processed
        }

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Transcribing"):
            result = fut.result()
            if result is None:
                continue
            utt_id, transcript, wav_path = result

            # Write Kaldi-format entries
            text_f.write(f"{utt_id} {transcript}\n")
            wavscp_f.write(f"{utt_id} {wav_path.resolve()}\n")
            utt2spk_f.write(f"{utt_id} {utt_id}\n")

    text_f.close()
    wavscp_f.close()
    utt2spk_f.close()
    print(f"Done. Kaldi files written under {out_dir}")


if __name__ == "__main__":
    main()
