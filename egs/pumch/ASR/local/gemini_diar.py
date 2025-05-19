import argparse
import json
from pathlib import Path

from google import genai
from tqdm import tqdm

GEMINI_PROMPT = """
You are given a medical doctor–patient reading session in which the patient has a cleft lip and palate resulting in an articulation disorder. During the session, the doctor asks the patient to read aloud a series of target sentences in Chinese. Because of the patient's speech impairment, an automatic speech recognition transcript may contain errors. Your job is to:
1. Process the given audio file. 
2. Perform speaker diarization to identify who spoke when, and automatic speech recognition (ASR) to transcribe the content.
3. Identify and correct any transcription mistakes by referencing the exact Chinese sentences below.
4. Return a clean, error-free transcript that matches what the patient should have said.

You should:
Return the output as a single JSON object containing a list of utterances. 
Do not merge two consecutive sentences. 
Ensure that diarization results for digits remain separate when the content involves counting digits. 
When encountering utterances with overlapping speech, create separate entries for each speaker's contribution, even if they overlap in time.
When encountering utterances containing entities, such as names or locations, ensure that part of the transcriptions are removed and the timestamps are adjusted accordingly.
Each utterance object in the list should include:
- "speaker_id": A unique identifier for the speaker (e.g., "Speaker 1"). always assign the same speaker id to the same speaker, and always make the doctor as "Speaker 1".
- "start_time": The start time of the utterance in seconds.
- "end_time": The end time of the utterance in seconds, calculated as the start time plus the duration of the utterance, make sure the end time lays after the speaker finishes speaking.
- "transcript": The transcribed text for that utterance. The transcribed text MUST be returned in Simplified Chinese 简体中文, including numbers (e.g., 十一，十二), punctuation.

Example format:
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

Provide only the JSON output, do not include any other text or explanation. Do not include any additional information, markdown syntax or metadata outside of the JSON format. The JSON should be valid and well-formed.

Do not merge two consecutive sentences. 
Ensure that diarization results for digits remain separate when the content involves counting digits. 
When encountering utterances with overlapping speech, create separate entries for each speaker's contribution, even if they overlap in time.
When encountering utterances containing entities, such as names or locations, ensure that part of the transcriptions are removed and the timestamps are adjusted accordingly.

Target Texts (in Chinese)
	从1数到10或从1数到20
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
    with open("./sk_token", "r") as f:
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
                model="gemini-2.5-pro-preview-05-06", contents=[GEMINI_PROMPT, audio_file_cli]
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
