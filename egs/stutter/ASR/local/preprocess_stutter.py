#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright    2024  Xiaomi Corp.        (authors: Zengrui Jin)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from pathlib import Path

import jieba
import paddle
from tqdm import tqdm

jieba.enable_paddle()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess transcript for stuttering speech challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kaldi-dir",
        type=str,
        help="the kaldi fmt directory for stuttering speech challenge",
    )
    return parser


def read_file_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    file_dict = {}
    for line in lines:
        key, value = line.split(" ", maxsplit=1)
        file_dict[key] = value
    return file_dict


def write_file_dict(file_dict, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for key, value in file_dict.items():
            f.write(f"{key} {value}\n")


# Copied from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
def is_cjk(character):
    """
    Python port of Moses' code to check for CJK character.

    >>> is_cjk(u'\u33fe')
    True
    >>> is_cjk(u'\uFE5F')
    False

    :param character: The character that needs to be checked.
    :type character: char
    :return: bool
    """
    return any(
        [
            start <= ord(character) <= end
            for start, end in [
                (4352, 4607),
                (11904, 42191),
                (43072, 43135),
                (44032, 55215),
                (63744, 64255),
                (65072, 65103),
                (65381, 65500),
                (131072, 196607),
            ]
        ]
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    kaldi_dir = Path(args.kaldi_dir)

    input_text = kaldi_dir / "text"
    text_preprocessed = kaldi_dir / "text.preprocessed"
    text_segment = kaldi_dir / "text.segment"
    utt2spk = kaldi_dir / "utt2spk"

    text_dict = read_file_dict(input_text)

    seg_lines = dict()
    preprocessed_lines = dict()
    for key, line in tqdm(text_dict.items(), desc="preprocessing text"):
        new_line = line.strip().upper().remove(",")
        seg_list = jieba.cut(new_line, use_paddle=True)
        seg_line = " ".join(seg_list)

        preprocessed_line = ""
        last_is_cjk = False
        for word in seg_list:
            if is_cjk(word[0]):
                if last_is_cjk:
                    preprocessed_line += word
                else:
                    preprocessed_line += f" {word}"
                last_is_cjk = True
            else:
                preprocessed_line += f" {word}"
                last_is_cjk = False
        preprocessed_lines[key] = new_line
        seg_lines[key] = seg_line

    write_file_dict(preprocessed_lines, text_preprocessed)

    write_file_dict(seg_lines, text_segment)

    utt2spk_lines = dict()
    for key in preprocessed_lines.keys():
        utt2spk_lines[key] = key

    write_file_dict(utt2spk_lines, utt2spk)
