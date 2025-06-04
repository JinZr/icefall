import argparse
import json
from pathlib import Path

from google import genai
from tqdm import tqdm

GEMINI_PROMPT = """
You are given a medical doctor-patient reading session in which the patient has a cleft lip and palate resulting in an articulation disorder. During the session, the doctor asks the patient to read aloud a series of target sentences in Chinese. Because of the patient's speech impairment, an automatic speech recognition transcript may contain errors. Your job is to:

TASKS  
1. Load and analyse the audio file.  
2. **Speaker diarization** - always label the doctor as `"Speaker 1"` and keep that ID consistent.  
3. **Automatic speech recognition (ASR).**

RULES  
1. **Verbatim transcription** - keep every stutter (“爸、爸爸”), repetition, filler (“呃”), and self-correction. Do not normalise or delete disfluencies.  
2. **Partial readings** - if the patient omits or stops mid-sentence, transcribe exactly what is audible. Never auto-complete.  
3. **Error-correction scope** - fix only obvious ASR typos for syllables that are clearly spoken. Do not “correct” mis-pronunciations or missing words.  
4. **Timing precision** -  
   • `start_time` is the true speech onset; `end_time` is the true offset (≤ 0.1 s of silence on either side).  
   • Use `[SILENCE]` segments or new utterances for pauses > 0.5 s.  
5. **Overlap** - create separate entries when speakers talk at the same time.  
6. **Privacy** - remove any names or locations that might appear.  
7. **Output** - return one valid JSON object and *nothing else*.

Example schema (illustrative only):
{
  "utterances": [
    {
      "speaker_id": "Speaker 1",
      "start_time": 0.5,
      "end_time": 3.2,
      "transcript": "Hello, this is the first segment."
    },
    {
      "speaker_id": "Speaker 2",
      "start_time": 3.5,
      "end_time": 6.8,
      "transcript": "And this is the second part from another speaker."
    }
    // ... more utterances
  ]
}

REFERENCE SENTENCES (for spell-checking only — **never copy words the patient does not actually say**):  
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

these sentences should stay as separate segments.
**REMEMBER - OUTPUT ONLY THE JSON; NEVER INSERT WORDS THAT WERE NOT HEARD.**
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="""Path to the directory containing the audio files""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Path to the directory where the output files will be saved""",
    )
    return parser.parse_args()


def main(args):
    audio_dir = args.audio_dir
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print("No audio files found in the specified directory.")
        return
    with open("./local/sk_token", "r") as f:
        sk_token = f.read().strip()
    for audio_file in tqdm(sorted(audio_files)):
        output_file = args.output_dir / (audio_file.stem + ".json")
        if output_file.exists():
            print(f"Output file {output_file} already exists. Skipping.")
            continue
        client = genai.Client(api_key=sk_token)
        audio_file_path = str(audio_file)

        audio_file_cli = client.files.upload(file=audio_file_path)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06",
                contents=[GEMINI_PROMPT, audio_file_cli],
            )
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
        response_text = response.text.replace("```json", "").replace("```", "")
        response_json = json.loads(response_text)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args=args)
