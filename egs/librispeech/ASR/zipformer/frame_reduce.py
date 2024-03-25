import torch
from torch import Tensor


def indexes_to_merge(num_frame: int, num_to_merge: int, **kwargs) -> Tensor:
    """
    Returns a Tensor containing indexes like [ 0 1 1 2 3 3 4 4 5 ], where, randomly,
    a certain proportion of the indexes are repeated twice and the remaining indexes
    are repeated once.

    Args:
      batch_size: the original batch size
      num_to_duplicate: the number of elements of the batch to duplicate
      kwargs: provided so you can add "device=something.device"
    Returns:
      a Tensor containing something like [ 0 1 1 2 3 3 4 4 5 ], with some indexes
      randomly duplicated.  At least one batch element will be duplicated.

      The caller can then do something like:
        x = torch.index_select(x, dim=0, index=ret)
      where ret is the return value of this function
    """
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


if __name__ == "__main__":
    feats = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        [
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [90, 100, 110, 120],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        [
            [100, 200, 300, 400],
            [500, 600, 700, 800],
            [900, 1000, 1100, 1200],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        [
            [1000, 2000, 3000, 4000],
            [5000, 6000, 7000, 8000],
            [9000, 10000, 11000, 12000],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        [
            [10000, 20000, 30000, 40000],
            [50000, 60000, 70000, 80000],
            [90000, 100000, 110000, 120000],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
    ]
    feats = torch.tensor(feats, dtype=torch.float32)
    print(feats.size())

    dup_indexes = indexes_to_merge(6, 2)
    print(dup_indexes)

    ret_feats = merge(feats, dup_indexes)
    print(ret_feats)
