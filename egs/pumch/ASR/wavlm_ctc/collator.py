from typing import Dict, List, Optional, Union

import torch
from transformers import Wav2Vec2Processor


class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Dynamically pad a batch of audio + label features so they can be fed to
        `Wav2Vec2ForCTC`. The function is robust to
        - `datasets.Audio` dicts (``{"array": np.ndarray, "sampling_rate": 16_000}``)
        - label IDs stored under either ``"labels"`` *or* ``"input_ids"``
        - optional nested dicts like ``{"input_ids": [...]}``.
        """

        input_features = []
        label_features = []

        for feat in features:
            # ─────────── Audio ───────────
            audio = feat["input_values"]
            # When the column is of type datasets.Audio it comes in as a dict
            if isinstance(audio, dict) and "array" in audio:
                audio = audio["array"]
            input_features.append({"input_values": audio})

            # ──────── Convert/unwrap labels ────────
            if "labels" in feat and feat["labels"] is not None:
                lab = feat["labels"]
            elif "input_ids" in feat:
                lab = feat["input_ids"]
            else:
                raise KeyError(
                    "Expected either 'labels' or 'input_ids' in feature but found none."
                )
            # 1) Unwrap nested dicts
            if isinstance(lab, dict):  # e.g. {"input_ids": [...]}
                lab = lab.get("input_ids", lab)

            # 2) If the label is still raw text, tokenize it to IDs
            if isinstance(lab, str):
                lab = self.processor.tokenizer(
                    lab,
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids

            # 3) If the label is a single int, put it into a list
            if isinstance(lab, int):
                lab = [lab]

            label_features.append({"input_ids": lab})

        # ─────────── Pad inputs ───────────
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # ─────────── Pad targets ───────────
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Replace pad token IDs by -100 so they are ignored by loss
        # labels = labels_batch["input_ids"]
        # labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch
