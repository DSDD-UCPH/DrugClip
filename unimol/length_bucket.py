# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Length-bucketing batch sampler for molecule encoding.

Attention cost is quadratic in the padded length of a batch. Grouping similar
``mol_len`` values cuts padding waste. Lengths are the already-computed
``mol_len`` field on the nested mol dataset (heavy-atom count before BOS/EOS).
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import Sampler
except ImportError:  # sampler still works; DataLoader helpers need torch
    torch = None
    Sampler = object


class LengthBucketBatchSampler(Sampler):
    """Yield lists of source indices, each list a batch of similar lengths.

    Buckets are contiguous ranges of sorted length. Within a bucket, indices
    are shuffled (optional) then packed into ``batch_size`` groups. The last
    batch of each bucket may be shorter.
    """

    def __init__(
        self,
        lengths,
        batch_size,
        bucket_boundaries=None,
        shuffle=False,
        seed=1,
        drop_last=False,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32).reshape(-1)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if bucket_boundaries is None:
            # Inclusive upper bounds. Drug-like heavy-atom counts cluster < 40.
            self.boundaries = np.array(
                [8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 256, 512, 10_000],
                dtype=np.int32,
            )
        else:
            self.boundaries = np.asarray(bucket_boundaries, dtype=np.int32)
        self._batches = self._build_batches()

    def _build_batches(self):
        n = int(self.lengths.shape[0])
        if n == 0:
            return []
        bucket_id = np.searchsorted(self.boundaries, self.lengths, side="left")
        rng = np.random.default_rng(self.seed)
        batches = []
        for b in np.unique(bucket_id):
            idx = np.flatnonzero(bucket_id == b)
            if self.shuffle:
                rng.shuffle(idx)
            else:
                # Stable order within bucket: increasing length, then index.
                order = np.lexsort((idx, self.lengths[idx]))
                idx = idx[order]
            for start in range(0, idx.size, self.batch_size):
                chunk = idx[start:start + self.batch_size]
                if self.drop_last and chunk.size < self.batch_size:
                    continue
                batches.append(chunk.astype(np.int64, copy=False))
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        for batch in self._batches:
            yield batch.tolist()

    def __len__(self):
        return len(self._batches)


def collect_mol_lens(mol_dataset, batch_size=1024, num_workers=0):
    """Read ``mol_len`` for every item. Returns int32 array of shape (N,)."""
    n = len(mol_dataset)

    def _len_collate(samples):
        return np.array([int(s["mol_len"]) for s in samples], dtype=np.int32)

    if num_workers and num_workers > 0:
        if torch is None:
            raise ImportError("num_workers>0 requires torch")
        loader = torch.utils.data.DataLoader(
            mol_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            collate_fn=_len_collate,
            drop_last=False,
        )
        chunks = [batch for batch in loader]
        return np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=np.int32)

    lengths = np.empty(n, dtype=np.int32)
    for i in range(n):
        item = mol_dataset[i]
        if not isinstance(item, dict) or "mol_len" not in item:
            raise KeyError("mol dataset items must contain mol_len")
        lengths[i] = int(item["mol_len"])
    return lengths


class CollateWithIndex:
    """Picklable collate that preserves source indices as batch['mol_index']."""

    def __init__(self, base_collate):
        self.base_collate = base_collate

    def __call__(self, samples):
        if torch is None:
            raise ImportError("CollateWithIndex requires torch")
        idxs = []
        stripped = []
        for s in samples:
            s = dict(s)
            idxs.append(int(s.pop("mol_index")))
            stripped.append(s)
        batch = self.base_collate(stripped)
        batch["mol_index"] = torch.tensor(idxs, dtype=torch.int64)
        return batch


def wrap_collate_with_index(base_collate):
    return CollateWithIndex(base_collate)


if torch is not None:
    _DatasetBase = torch.utils.data.Dataset
else:
    _DatasetBase = object


class IndexedDataset(_DatasetBase):
    """Attach ``mol_index`` so a BatchSampler's source ids survive collate."""

    def __init__(self, base):
        self.base = base
        self.collater = wrap_collate_with_index(base.collater)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if isinstance(item, dict):
            out = dict(item)
        else:
            out = {"_item": item}
        out["mol_index"] = int(index)
        return out
