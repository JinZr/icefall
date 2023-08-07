# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from cif import CifMiddleware
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos, make_pad_mask


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        cif: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_quantity_loss: bool = True,
        use_decoder_ce_loss: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.cif = cif
        self.use_quantity_loss = use_quantity_loss
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=1)

        self.use_transducer = use_transducer

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
        self.use_decoder_ce_loss = use_decoder_ce_loss
        if use_decoder_ce_loss:
            self.decoder_ce_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(decoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 2,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        sos_y = add_sos(y, sos_id=0)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=0)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)
        
        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens
        
        decoder_ce_loss = torch.tensor(0.0).cuda()
        if self.use_decoder_ce_loss:
            decoder_ce_output = self.decoder_ce_output(decoder_out)
            decoder_ce_loss = self.ce_loss(
                decoder_ce_output.view(-1, self.joiner.vocab_size),
                F.pad(y_padded, (0, 1), "constant", 0).flatten(),
            )


        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=0,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        proj_am = self.joiner.encoder_proj(encoder_out)
        proj_lm = self.joiner.decoder_proj(decoder_out)

        cif_out_dict = self.cif(
            proj_am, make_pad_mask(encoder_out_lens), y_lens.cuda() + 1
        )
        # print(y_lens)
        cif_out, cif_out_padding_mask, cif_out_lens, quantity_out, cif_weight = (
            cif_out_dict["cif_out"],
            cif_out_dict["cif_out_padding_mask"],
            cif_out_dict["cif_out_lens"],
            cif_out_dict["quantity_out"],
            cif_out_dict["cif_weight"],
        )

        expanded_ranges = nn.ZeroPad2d(padding=(0, 0, 0, 1))(ranges)
        shifted_ranges = nn.ZeroPad2d(padding=(0, 0, 1, 0))(ranges)
        breaking_points = (
            ~(expanded_ranges == shifted_ranges).int().sum(-1).bool()[:, 1:]
        )  # shape B x T

        batch_size = encoder_out.size(0)
        max_length = encoder_out.size(1)
        encoder_embed_dim = encoder_out.size(2)
        encoder_out_mask = make_pad_mask(encoder_out_lens)
        # padding_start_id = not_padding_mask.sum(-1)  # shape B
        padding_start_id = ~encoder_out_mask.sum(-1)  # shape B

        # Calculate the quantity loss
        qtt_loss = torch.tensor(0.0).cuda()
        if self.use_quantity_loss:
            cif_weight_cumsum = torch.cumsum(cif_weight, dim=1)
            for i in range(batch_size):
                breaking_points_index = torch.nonzero(breaking_points[i]).squeeze(
                    1
                )  # [ num_non_zero_pts ]
                selected_cif_w_cumsum = cif_weight_cumsum[i, breaking_points_index]
                shifted_selected_cif_w_cumsum = torch.nn.ConstantPad1d((1, 0), 0)(
                    selected_cif_w_cumsum
                )
                extended_selected_cif_w_cumsum = torch.nn.ConstantPad1d((0, 1), 0)(
                    selected_cif_w_cumsum
                )
                diff_selected_cif_w_cumsum = (
                    extended_selected_cif_w_cumsum - shifted_selected_cif_w_cumsum
                )[:-1]
                per_utter_qtt_loss = torch.abs(1 - diff_selected_cif_w_cumsum).sum()
                qtt_loss += per_utter_qtt_loss
            # target_lengths_for_qtt_loss = (
            #     y_lens.cuda() + 1
            # )  # Lengths after adding eos token, [B]
            # qtt_loss = torch.abs(quantity_out - target_lengths_for_qtt_loss).sum()

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        # am_pruned, lm_pruned = k2.do_rnnt_pruning(
        #     am=self.joiner.encoder_proj(encoder_out),
        #     lm=self.joiner.decoder_proj(decoder_out),
        #     ranges=ranges,
        # )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        # logits = self.joiner(am_pruned, lm_pruned, project_input=False)
        # print(cif_out.shape)
        # print(y_padded.shape)
        logits = self.joiner(cif_out, proj_lm, project_input=False)

        # with torch.cuda.amp.autocast(enabled=False):
        #     pruned_loss = k2.rnnt_loss_pruned(
        #         logits=logits.float(),
        #         symbols=y_padded,
        #         ranges=ranges,
        #         termination_symbol=blank_id,
        #         boundary=boundary,
        #         reduction="sum",
        #     )

        ce_loss = self.ce_loss(
            logits.view(-1, self.joiner.vocab_size),
            F.pad(y_padded, (0, 1), "constant", 0).flatten(),
        )  # B x T x vocab_size

        # return simple_loss, pruned_loss
        # return simple_loss, pruned_loss, qtt_loss
        return simple_loss, decoder_ce_loss, ce_loss, qtt_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            # simple_loss, pruned_loss = self.forward_transducer(
            #     encoder_out=encoder_out,
            #     encoder_out_lens=encoder_out_lens,
            #     y=y.to(x.device),
            #     y_lens=y_lens,
            #     prune_range=prune_range,
            #     am_scale=am_scale,
            #     lm_scale=lm_scale,
            # )
            # simple_loss, pruned_loss, qtt_loss = self.forward_transducer(
            #     encoder_out=encoder_out,
            #     encoder_out_lens=encoder_out_lens,
            #     y=y.to(x.device),
            #     y_lens=y_lens,
            #     prune_range=prune_range,
            #     am_scale=am_scale,
            #     lm_scale=lm_scale,
            # )
            simple_loss, decoder_ce_loss, ce_loss, qtt_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            qtt_loss = torch.empty(0)
            ce_loss = torch.empty(0)
            decoder_ce_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)


        return simple_loss, decoder_ce_loss, ce_loss, ctc_loss, qtt_loss
