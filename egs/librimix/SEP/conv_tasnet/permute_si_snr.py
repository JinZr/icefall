# wujian@2018
"""
SI-SNR(scale-invariant SNR/SDR) measure of speech separation
"""

from itertools import permutations

import torch


def sisnr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape
            )
        )
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        * s_zm
        / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    )
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def permute_si_snr(ests, refs, batch_size):
    # ests: spks x n x S
    # refs: spks x n x S
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum([sisnr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(
            permute
        )

    sisnr_mat = torch.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)

    # si-snr
    return -torch.sum(max_perutt) / batch_size
