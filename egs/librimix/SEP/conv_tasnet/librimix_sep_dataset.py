# Copyright      2021  Piotr Żelasko
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


import logging
from typing import Dict, Tuple

import torch
from lhotse import CutSet
from lhotse.cut import Cut, MultiCut
from lhotse.dataset import PreMixedSourceSeparationDataset


class LibriMixSpeechSeparationDataset(PreMixedSourceSeparationDataset):
    """
    .. warning: Speech separation datasets are not yet updated to use the new Lhotse's sampling mechanism.

    An abstract base class, implementing PyTorch Dataset for the source separation task.
    It's created from two CutSets - one provides the audio cuts for the sources, and the other one the audio cuts for
    the signal mix. When queried for data samples, it returns a dict of:

    .. code-block::

        {
            'sources': (N x T x F) tensor,
            'mixture': (T x F) tensor,
        }
    """

    def __init__(
        self,
        sources_set: CutSet,
        mixtures_set: CutSet,
        chunk_duration: float = 2.0,
    ):
        super().__init__(
            sources_set=sources_set,
            mixtures_set=mixtures_set,
        )
        # self.sources_set = sources_set
        # self.mixtures_set = mixtures_set

        self.chunk_duration = chunk_duration

        self.sources_set_chunks = self.sources_set.cut_into_windows(
            duration=chunk_duration
        )
        self.mixtures_set_chunks = self.mixtures_set.cut_into_windows(
            duration=chunk_duration
        )
        self.cut_ids = list(self.mixtures_set_chunks.ids)
        logging.info(f"Created a dataset with {len(self)} cuts.")

    def _obtain_mixture(self, cut_id: str) -> Tuple[Cut, MultiCut]:
        mixture_cut = self.mixtures_set.cuts[cut_id]
        source_cut = self.sources_set.cuts[cut_id]
        return mixture_cut, source_cut

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut_id = self.cut_ids[idx]
        mixture_cut, source_cut = self._obtain_mixture(cut_id=cut_id)

        mixture = torch.from_numpy(mixture_cut.load_audio())
        sources = torch.from_numpy(source_cut.load_audio())

        return {
            "spk0": sources[0],
            "spk1": sources[1],
            "mixture": mixture[0],
        }
