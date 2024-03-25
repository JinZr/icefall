import torch
from torch import Tensor, nn


def indexes_to_pool(batch_size: int, num_to_pool: int, **kwargs) -> Tensor:
    return torch.randperm(batch_size, **kwargs)[:num_to_pool].sort(dim=0)[0]


def indexes_with_dups(num_frame: int, num_to_merge: int, **kwargs) -> Tensor:

    return torch.randperm(num_frame - 1, **kwargs)[:num_to_merge].sort(dim=0)[0]


def merge(feats: Tensor, dup_indexes: Tensor) -> Tensor:

    # feats: [batch_size, num_frames, num_features]
    to_merge_indexes = dup_indexes + 1
    print(to_merge_indexes)

    left_frames = feats.index_select(dim=1, index=dup_indexes)
    right_frames = feats.index_select(dim=1, index=to_merge_indexes)

    pooled_frames = (left_frames + right_frames) / 2
    print(pooled_frames)
    ret_feats = feats.index_copy(dim=1, index=dup_indexes, source=pooled_frames)
    ret_feats.index_copy_(dim=1, index=to_merge_indexes, source=pooled_frames)

    return ret_feats


class FramePool(nn.Module):
    def __init__(self, ratio: float = 0.25):
        super(FramePool, self).__init__()

        self.ratio = ratio
        self.pool_layer = nn.AvgPool1d(
            kernel_size=2, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, feats: Tensor, max_len: int) -> Tensor:
        batch_size, _, _ = feats.size()

        num_to_pool = int(batch_size * self.ratio)
        ind_to_pool = indexes_to_pool(batch_size, num_to_pool, device=feats.device)
        # print(ind_to_pool)

        feats_to_pool = feats.index_select(dim=0, index=ind_to_pool)
        feats_pooled = self.pool_layer(feats_to_pool.permute(0, 2, 1)).permute(0, 2, 1)
        # print("feats_pooled.size():", feats_pooled.size())

        feats_dup = torch.repeat_interleave(feats_pooled, repeats=2, dim=1)[
            :, :max_len, :
        ]
        # print("feats_dup.size():", feats_dup.size())
        # print("feats.size():", feats.size())

        ret_feats = feats.index_copy(dim=0, index=ind_to_pool, source=feats_dup)

        return ret_feats


if __name__ == "__main__":
    feats = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [9, 10, 11, 12],
        ],
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [9, 10, 11, 12],
        ],
        [
            [100, 200, 300, 400],
            [500, 600, 700, 800],
            [900, 1000, 1100, 1200],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [9, 10, 11, 12],
        ],
        [
            [1000, 2000, 3000, 4000],
            [5000, 6000, 7000, 8000],
            [9000, 10000, 11000, 12000],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [9, 10, 11, 12],
        ],
        [
            [10000, 20000, 30000, 40000],
            [50000, 60000, 70000, 80000],
            [90000, 100000, 110000, 120000],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [9, 10, 11, 12],
        ],
    ]
    pool_layer = nn.AvgPool1d(kernel_size=2, stride=2)
    feats = torch.tensor(feats, dtype=torch.float32)
    # print(feats.size())

    # dup_indexes = indexes_with_dups(6, 2)
    # print(dup_indexes)

    # ret_feats = merge(feats, dup_indexes)
    # print(ret_feats)
    # print(pool_layer(feats.permute(0, 2, 1)).permute(0, 2, 1).repeat(1, 2, 1).shape)
    # print(
    #     torch.repeat_interleave(
    #         pool_layer(feats.permute(0, 2, 1)).permute(0, 2, 1), 2, dim=1
    #     )
    # )

    frame_pool = FramePool()
    print(frame_pool(feats, max_len=7))
