# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared GPU encode helpers for shard-index build / Phase 0 / rerank."""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore

from unimol.length_bucket import (
    IndexedDataset,
    LengthBucketBatchSampler,
    collect_mol_lens,
)

logger = logging.getLogger(__name__)


def load_fold_encoder(task, model, fold, fold_version, use_cuda, use_fp16):
    ckpts = task._fold_checkpoints(fold_version)
    fold = int(fold)
    if fold < 0 or fold >= len(ckpts):
        raise ValueError(f"fold {fold} out of range for {fold_version}")
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpts[fold])
    model.load_state_dict(state["model"], strict=False)
    return task._snapshot_mol_encoder(model, use_cuda, use_fp16)


def load_all_fold_encoders(task, model, fold_version, use_cuda, use_fp16):
    ckpts = task._fold_checkpoints(fold_version)
    out = {}
    for fold, ckpt in enumerate(ckpts):
        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
        model.load_state_dict(state["model"], strict=False)
        out[fold] = task._snapshot_mol_encoder(model, use_cuda, use_fp16)
    return out


def make_bucketed_loader(task, mol_dataset, bsz, use_cuda, use_length_bucket=True):
    collate_fn = task._resolve_collater(mol_dataset)
    indexed = IndexedDataset(mol_dataset)
    kwargs = dict(task._mol_dataloader_kwargs(use_cuda))
    if use_length_bucket:
        num_workers = int(kwargs.get("num_workers") or 0)
        logger.info("collecting mol_len for length bucketing")
        lengths = collect_mol_lens(
            mol_dataset, batch_size=max(256, int(bsz)), num_workers=num_workers
        )
        sampler = LengthBucketBatchSampler(
            lengths, batch_size=int(bsz), shuffle=False, seed=1
        )
        logger.info(
            f"length buckets: {len(sampler)} batches, "
            f"len min/median/max={int(lengths.min())}/{int(np.median(lengths))}/{int(lengths.max())}"
        )
        return torch.utils.data.DataLoader(
            indexed,
            batch_sampler=sampler,
            collate_fn=indexed.collater,
            **{k: v for k, v in kwargs.items() if k != "shuffle"},
        )
    return torch.utils.data.DataLoader(
        indexed,
        batch_size=int(bsz),
        shuffle=False,
        collate_fn=indexed.collater,
        **kwargs,
    )


def _decode_smi_name(name):
    if hasattr(name, "decode"):
        name = name.decode("utf-8", "replace")
    return str(name)


def encode_fold_embeddings(
    task,
    mol_model,
    mol_project,
    mol_dataset,
    use_cuda,
    bsz,
    use_length_bucket=True,
    run_label="encode",
    collect_names=False,
):
    """Return (n_mols, d) float32 embeddings in source-index order.

    If collect_names is True, return (embs, names) instead.
    """
    n = len(mol_dataset)
    embs = None
    names = [None] * n if collect_names else None
    filled = np.zeros(n, dtype=np.bool_)
    loader = make_bucketed_loader(
        task, mol_dataset, bsz, use_cuda, use_length_bucket=use_length_bucket
    )
    with torch.inference_mode():
        for sample in tqdm(loader, desc=run_label):
            idx = sample["mol_index"].cpu().numpy().astype(np.int64, copy=False)
            if use_cuda:
                sample = unicore.utils.move_to_cuda(sample)
            vec = task._encode_mol_batch_tensor(mol_model, mol_project, sample)
            arr = vec.float().detach().cpu().numpy()
            if embs is None:
                embs = np.empty((n, arr.shape[-1]), dtype=np.float32)
            embs[idx] = arr
            filled[idx] = True
            if names is not None:
                smi = sample.get("smi_name")
                if smi is not None:
                    for j, name in enumerate(smi):
                        names[int(idx[j])] = _decode_smi_name(name)
    if embs is None:
        raise RuntimeError("encode produced no batches")
    if not np.all(filled):
        missing = int(np.count_nonzero(~filled))
        raise RuntimeError(f"encode missed {missing}/{n} molecules")
    if collect_names:
        return embs, names
    return embs


def score_folds_streaming(
    task,
    fold_encoders,
    pocket_reps_by_fold,
    mol_dataset,
    use_cuda,
    bsz,
    gate_fold=None,
    use_length_bucket=False,
    run_label="score",
):
    """One DataLoader pass. Returns (mean_scores, gate_scores, names).

    mean_scores / gate_scores are float32 (n_pock, n_mols).
    """
    n_mols = len(mol_dataset)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    pocket_t = {
        fold: torch.from_numpy(np.ascontiguousarray(reps)).to(device).float()
        for fold, reps in pocket_reps_by_fold.items()
    }
    n_pock = next(iter(pocket_reps_by_fold.values())).shape[0]
    n_folds = len(fold_encoders)
    mean_scores = np.empty((n_pock, n_mols), dtype=np.float32)
    gate_scores = None
    if gate_fold is not None:
        gate_scores = np.empty((n_pock, n_mols), dtype=np.float32)
    names = [None] * n_mols
    loader = make_bucketed_loader(
        task, mol_dataset, bsz, use_cuda, use_length_bucket=use_length_bucket
    )
    with torch.inference_mode():
        for sample in tqdm(loader, desc=run_label):
            idx = sample["mol_index"].cpu().numpy().astype(np.int64, copy=False)
            if use_cuda:
                sample = unicore.utils.move_to_cuda(sample)
            mean = None
            for fold, (mol_model, mol_project) in fold_encoders.items():
                emb = task._encode_mol_batch_tensor(mol_model, mol_project, sample)
                score = pocket_t[fold] @ emb.float().t()
                score_np = score.detach().cpu().numpy().astype(np.float32, copy=False)
                if mean is None:
                    mean = score_np
                else:
                    mean = mean + score_np
                if gate_fold is not None and int(fold) == int(gate_fold):
                    gate_scores[:, idx] = score_np
            mean = mean / float(n_folds)
            mean_scores[:, idx] = mean
            smi = sample.get("smi_name")
            if smi is not None:
                for j, name in enumerate(smi):
                    if hasattr(name, "decode"):
                        name = name.decode("utf-8", "replace")
                    names[int(idx[j])] = str(name)
    return mean_scores, gate_scores, names
