#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Exact all-fold rerank of a quantized-index shortlist.

Groups survivors by shard, loads them from packed LMDBs via
``_subset_mol_dataset``, re-encodes every fold (including fold 4; quantized
gate scores are never reused), and ranks with ``_ensemble_rank_mols``.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch
from unicore import distributed_utils, options
from unicore import tasks

from unimol.index_encode import load_all_fold_encoders, score_folds_streaming
from unimol.index_manifest import load_manifest
from unimol.tasks._drugclip_rank import _ensemble_rank_mols

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.rerank_shortlist")


def load_shortlist(path):
    data = np.load(path)
    return {
        "global_idx": np.asarray(data["global_idx"], dtype=np.int64),
        "metric": np.asarray(data["metric"], dtype=np.float32),
        "shard_id": np.asarray(data["shard_id"], dtype=np.int32),
        "local_idx": np.asarray(data["local_idx"], dtype=np.int32),
    }


def _group_by_shard(shortlist):
    sid = shortlist["shard_id"]
    loc = shortlist["local_idx"]
    order = np.argsort(sid, kind="stable")
    sid_s = sid[order]
    loc_s = loc[order]
    groups = []
    start = 0
    n = int(sid_s.size)
    while start < n:
        cur = int(sid_s[start])
        end = start + 1
        while end < n and int(sid_s[end]) == cur:
            end += 1
        groups.append((cur, loc_s[start:end].astype(np.int64)))
        start = end
    return groups


def rerank_shortlist(
    task,
    model,
    shortlist_path,
    pocket_path,
    manifest,
    use_cuda,
    use_fp16,
    bsz,
    fold_version,
    use_length_bucket=False,
    write_path=None,
    write_k=None,
):
    short = load_shortlist(shortlist_path)
    shards = {int(s["id"]): s for s in manifest["shards"]}
    pocket_dataset = task.load_pockets_dataset(pocket_path)
    pocket_data = task._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)
    ckpts = task._fold_checkpoints(fold_version)
    pocket_reps = task._cache_pocket_reps_by_fold(
        model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, True
    )
    encoders = load_all_fold_encoders(task, model, fold_version, use_cuda, use_fp16)

    mean_parts = []
    names = []
    for sid, local in _group_by_shard(short):
        rec = shards[int(sid)]
        mol_dataset = task.load_mols_dataset(
            rec["lmdb"], "atoms", "coordinates", readahead=False
        )
        subset = task._subset_mol_dataset(mol_dataset, local.tolist())
        logger.info(f"rerank shard {sid}: {len(subset)} mols from {rec['lmdb']}")
        mean, _, nms = score_folds_streaming(
            task,
            encoders,
            pocket_reps,
            subset,
            use_cuda,
            bsz,
            gate_fold=None,
            use_length_bucket=use_length_bucket,
            run_label=f"rerank shard {sid}",
        )
        mean_parts.append(mean)
        names.extend(nms)
        task._close_mol_lmdb_envs(mol_dataset)

    if not mean_parts:
        raise RuntimeError("empty shortlist")
    mean_all = np.concatenate(mean_parts, axis=1)
    # Already the fold mean; pass as a one-fold list so z-score/max match retrieval.
    metric = _ensemble_rank_mols([mean_all], fold_version)
    order = np.argsort(metric, kind="stable")[::-1]
    if write_k is not None:
        order = order[: max(1, int(write_k))]
    if write_path:
        parent = os.path.dirname(os.path.abspath(write_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(write_path, "w") as f:
            for i in order:
                f.write(f"{names[int(i)]},{float(metric[int(i)])}\n")
        logger.info(f"wrote {write_path} ({order.size} rows)")
    return metric, names, order


def main(args):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()
    model.eval()
    man = load_manifest(args.manifest)
    fold_version = man.get("fold_version", args.fold_version)
    bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else 256
    out = args.out_path or os.path.splitext(args.shortlist)[0] + ".rerank.csv"
    rerank_shortlist(
        task,
        model,
        args.shortlist,
        args.pocket_path,
        man,
        use_cuda,
        use_fp16,
        bsz,
        fold_version,
        use_length_bucket=bool(args.length_bucket),
        write_path=out,
        write_k=args.write_k if args.write_k and args.write_k > 0 else None,
    )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--shortlist", type=str, required=True)
    parser.add_argument("--pocket-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="")
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--length-bucket", type=int, default=0)
    parser.add_argument("--write-k", type=int, default=0)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
