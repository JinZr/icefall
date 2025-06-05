import argparse
import json
import time
from pathlib import Path

from google import genai
from tqdm import tqdm

GEMINI_PROMPT = """
You are given a **synchronised video clip** (with its audio track) of a medical doctor–patient reading session.  
The patient has a cleft lip and palate that causes articulation disorders.  
Your job is to produce a diarised, verbatim transcript that faithfully captures what is actually spoken, using **both audio and visual cues (lip movements, speaker presence)** to maximise accuracy.

TASKS  
1. Load and analyse the audio and the corresponding video frames.  
2. **Speaker diarization** – always label the doctor as `"Speaker 1"` and keep that ID consistent. Use both voice characteristics and visual speaker detection to decide who is talking, especially during overlaps.  
3. **Automatic speech recognition (ASR)** – fuse audio and lip‑reading information to obtain the most faithful verbatim transcript.  
4. **Cross‑modal verification** – when the audio is ambiguous (e.g., noise, mis‑pronunciation) rely on mouth shapes and context; when lip motion and audio disagree, prefer what is unequivocally *spoken* (never auto‑complete the reference sentence).

RULES  
1. **Verbatim transcription** – retain every stutter (“爸、爸爸”), repetition, filler (“呃”), and self‑correction. Do not normalise or delete disfluencies.  
2. **Partial readings** – if the patient omits or stops mid‑sentence, transcribe exactly what is audible/visible. Never auto‑complete.  
3. **Error‑correction scope** – fix only obvious ASR typos for syllables and mis‑pronunciations that are clearly spoken; do **not** “correct” missing words.  
4. **Timing precision** –  
   • `start_time` is the true speech onset; `end_time` is the true offset (≤ 0.1 s of silence on either side).  
5. **Overlap** – create separate entries when speakers talk at the same time.  
6. **Privacy & visual content** – do not describe personal appearance; remove utterances that contain any names or locations that might appear in speech or on screen.  
7. **Output** – return one valid JSON object and *nothing else*.

Example schema (illustrative only):
{
  "utterances": [
    {
      "speaker_id": "Speaker 1",
      "start_time": 0.5,
      "end_time": 3.2,
      "transcript": "您好，这是第一段。"
    },
    {
      "speaker_id": "Speaker 2",
      "start_time": 3.5,
      "end_time": 6.8,
      "transcript": "这是另一位说话者的部分。"
    }
    // ... more utterances
  ]
}

REFERENCE SENTENCES (for spell‑checking only — **never copy words the patient does not actually say**):  
    一二三四五六七八九十  
    一二三四五六七八九十十一十二十三十四十五十六十七十八十九二十  
    爸爸跑步  
    弟弟踢皮球  
    哥哥喝可乐  
    头发飞飞  
    谢谢姐姐  
    妈妈买牛奶  
    猴子喜欢香蕉  
    大象喜欢草莓  
    长颈鹿喜欢山楂树  
    早饭在桌子上  
    奇怪而有趣的旗  
    姥姥的榴莲  
    叔叔的老师  
    长长的长城  
    炒菜菜  
    奶奶买柠檬  
    一起去爬坡
    宝宝带板凳
    贝贝唱支歌
    妈妈牛牛毛毛猫
    妈妈模样美
    牛牛没眉毛
    他去无锡市
    我到黑龙江
**REMEMBER – RETURN ONLY THE JSON; NEVER INSERT WORDS THAT ARE NOT HEARD OR SEEN.**
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="""Path to the directory containing the video files""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Path to the directory where the output files will be saved""",
    )
    return parser.parse_args()


def wait_until_active(client, file_obj, poll_interval: int = 5, timeout: int = 600):
    """
    Poll the file status until it becomes ACTIVE or raise a RuntimeError
    after the timeout.

    Args:
        client: genai.Client instance.
        file_obj: The File object returned by `client.files.upload`.
        poll_interval: Seconds between status checks.
        timeout: Maximum seconds to wait before giving up.

    Returns:
        The refreshed File object in ACTIVE state.
    """
    waited = 0
    while not getattr(file_obj, "state", None) or file_obj.state.name != "ACTIVE":
        if waited >= timeout:
            raise RuntimeError(
                f"File {getattr(file_obj, 'name', 'UNKNOWN')} "
                f"did not become ACTIVE within {timeout} s."
            )
        time.sleep(poll_interval)
        waited += poll_interval
        file_obj = client.files.get(name=file_obj.name)
    return file_obj


def main(args):
    video_dir = args.video_dir
    # Collect videos with common extensions, case‑insensitive
    video_files = []
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        video_files.extend(video_dir.glob(pattern))
    if not video_files:
        print("No .mp4 or .mov video files found in the specified directory.")
        return
    with open("./local/sk_token", "r") as f:
        sk_token = f.read().strip()
    client = genai.Client(api_key=sk_token)

    for video_file in tqdm(sorted(video_files)):
        output_file = args.output_dir / (video_file.stem + ".json")
        if output_file.exists():
            print(f"Output file {output_file} already exists. Skipping.")
            continue

        upload_path = video_file
        print(f"Processing {video_file}...")
        video_file_cli = client.files.upload(file=upload_path)
        try:
            # Wait until the uploaded video is fully processed
            video_file_cli = wait_until_active(client, video_file_cli)
            response = client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06",
                contents=[GEMINI_PROMPT, video_file_cli],
            )
        except RuntimeError as e:
            print(f"Skipping {video_file} – {e}")
            continue
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
        response_text = response.text.replace("```json", "").replace("```", "")
        response_json = json.loads(response_text)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
