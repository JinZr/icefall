import argparse
import json
import time
from pathlib import Path

import cv2
from google import genai
from tqdm import tqdm

GEMINI_PROMPT = """
You are given a synchronised video clip (with its audio track) of a medical doctor-patient reading session. The patient has a cleft lip and palate that causes articulation disorders. Your job is to produce a diarised, verbatim transcript that faithfully captures what is actually spoken, using both audio and visual cues (lip movements, speaker presence) to maximise accuracy.

TASKS
	1.	Load and analyse the audio and corresponding video frames.
	2.	Speaker diarization - always label the doctor as "Speaker 1" and keep that ID consistent. Use both voice characteristics and visual speaker detection to decide who is talking, especially during overlaps.
	3.	Automatic speech recognition (ASR) - fuse audio and lip-reading information to obtain the most faithful verbatim transcript.
	4.	Cross-modal verification - when the audio is ambiguous (e.g., noise, mispronunciation), rely on mouth shapes and context; when lip motion and audio disagree, prefer what is unequivocally spoken (never auto-complete the reference sentence).

RULES
	1.	Verbatim transcription - retain every stutter (“爸、爸爸”), repetition, filler (“呃”), and self-correction. Do not normalise or delete disfluencies.
	2.	Partial readings - if the patient omits or stops mid-sentence, transcribe exactly what is audible/visible. Never auto-complete.
	3.	Error-correction scope - fix only obvious ASR typos for syllables and mispronunciations that are clearly spoken; do not “correct” missing words.
	4.	Timing precision constraints -
	•	start_time and end_time must precisely reflect speech onset and offset, respectively.
	•	The duration of an utterance (end_time - start_time) must always reflect realistic speech intervals.
	•	Ensure no single utterance exceeds a maximum duration of 10 seconds, unless it genuinely and visually represents continuous speech without pauses.
	•	If speech is discontinuous or has noticeable pauses (>0.3 s), create separate utterances accordingly.
	5.	Overlap - create separate entries when speakers talk simultaneously, each with accurate timestamps.
	6.	Privacy & visual content - do not describe personal appearance; remove utterances containing any names or locations visible or audible.
	7.	Output - return one valid JSON object and nothing else.
    8.  The Speaker 2 is asked to repeat some utterances that Speaker 1 has said. You can use transcript of utterances spoken by Speaker 1 to ensure that transcripts of Speaker 2 utterances are correct".
    9.  Make the transcript of counting utterances consecutive, do not split them into multiple utterances if it's not interrupted. For example, if Speaker 2 says "一二三四五六七八九十" or "一二三四五六七八九十十一十二十三十四十五十六十七十八十九二十", it should be one utterance with the full transcript, not split into individual numbers.

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


REFERENCE SENTENCES (for spell-checking only — never insert words not explicitly spoken):
一二三四五六七八九十, 一二三四五六七八九十十一十二十三十四十五十六十七十八十九二十, 爸爸跑步, 弟弟踢皮球, 哥哥喝可乐, 头发飞飞, 谢谢姐姐, 妈妈买牛奶, 猴子喜欢香蕉, 大象喜欢草莓, 长颈鹿喜欢山楂树, 早饭在桌子上, 奇怪而有趣的旗, 姥姥的榴莲, 叔叔的老师, 长长的长城, 炒菜菜, 奶奶买柠檬, 一起去爬坡, 宝宝带板凳, 贝贝唱支歌, 妈妈牛牛磨麦苗, 妈妈模样美, 牛牛眉眼浓. 
早晨推开前窗蜜蜂嗡嗡叫在花丛中飞舞, 大姐做饭二哥准备行囊我学妈妈认真收拾衣裳老爸永远是司机驾车去山村游玩, 我跑进羊群怀抱抚摸小羊云雾缭绕, 暖阳洒在村庄的水波中, 老翁在岸旁的柳树下花海前, 美丽的鸟儿在枝头愉快地歌唱, 微风吹过远处雄山隐约可见令人思接千载.
他去无锡市, 我到黑龙江, 瑞雪初融迎新春, 彩云追月照家门, 对联高悬祈安康, 福字倒挂纳吉祥, 鞭炮齐鸣多喜乐, 烟花绽放祛忧愁, 爹娘堂下拜翁婆, 儿女院中试新装, 最喜兄弟梦正美, 快船撒网钓鱼虾.

REMEMBER - RETURN ONLY THE JSON; NEVER INSERT WORDS THAT ARE NOT HEARD OR SEEN. ENSURE ALL TIMESTAMPS ARE ACCURATE, CONTINUOUS, AND REFLECT REALISTIC SPEECH INTERVALS. THE INTERVAL BETWEEN TWO CONSECUTIVE UTTERANCES MUST NOT EXCEED 2 SECONDS UNLESS THERE IS A REAL PAUSE IN SPEECH."""


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
        # Calculate video duration
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0
        cap.release()
        duration = round(duration, 2)
        # Append duration to prompt
        prompt_with_duration = (
            GEMINI_PROMPT
            + f"\nVideo duration: {duration} seconds, ensure that the timestamps in the output are accurate and reflect this duration. DO NOT BUMP THE TIMESTAMPS FROM AROUND 50 SECONDS TO MORE THAN 100 SECONDS."
        )

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
                model="gemini-2.5-pro",
                contents=[prompt_with_duration, video_file_cli],
            )
        except RuntimeError as e:
            print(f"Skipping {video_file} - {e}")
            continue
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
        response_text = response.text.replace("```json", "").replace("```", "")
        response_json = json.loads(response_text)
        response_json["speaker_id"] = video_file.stem.split(" ")[
            -1
        ]  # Ensure speaker ID is set
        with open(output_file, "w", encoding="utf-8") as f:
            print(output_file)
            json.dump(response_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
