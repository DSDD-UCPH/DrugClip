#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sequential 4-bit code scan over the shard index.

Encodes pockets for the session, builds per-pocket median/MAD from the
manifest's fixed anchor sample (dequantized codes — no extra LMDB read),
then streams every shard.codes.npy into one running top-k heap per target
pocket-LMDB on the max-over-pockets z-score.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from unicore import distributed_utils, options
from unicore import tasks

from unimol.index_manifest import load_manifest, split_global_indices, write_shortlist
from unimol.tasks._drugclip_rank import (
    _RunningTopK,
    _batch_gate_metric,
    _robust_pocket_anchors,
)
from unimol.turboquant import TurboQuant

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.scan_index")


def _target_stem(path):
    return os.path.splitext(os.path.basename(os.path.abspath(path)))[0]


def _dequant_chunk_torch(codes_u8, codebook, device, tq=None):
    """Dequantize a codes chunk on ``device``.

    Production default is 4-bit packed nibbles. Other bit widths use
    ``TurboQuant`` packing metadata so ablation scans stay bit-correct.
    """
    t = torch.from_numpy(np.ascontiguousarray(codes_u8)).to(device, non_blocking=True)
    bits = 4 if tq is None else int(tq.bits)
    packed = True if tq is None else bool(tq.packed)
    dim = int(tq.dim) if tq is not None else int(t.shape[1]) * 2
    if bits == 4 and packed:
        hi = torch.bitwise_right_shift(t, 4).to(torch.long)
        lo = torch.bitwise_and(t, 0x0F).to(torch.long)
        idx = torch.stack((hi, lo), dim=-1).reshape(t.shape[0], dim)
        return codebook[idx]
    if packed:
        from unimol.turboquant import unpack_indices

        idx_np = unpack_indices(np.ascontiguousarray(codes_u8), dim, bits)
        idx = torch.from_numpy(np.ascontiguousarray(idx_np)).to(
            device, dtype=torch.long, non_blocking=True
        )
        return codebook[idx]
    idx = t.to(torch.long)
    return codebook[idx]


def gather_anchor_codes(manifest, tq):
    anchors = np.asarray(manifest.get("anchor_indices") or [], dtype=np.int64)
    if anchors.size == 0:
        raise ValueError("manifest has no anchor_indices")
    grouped = split_global_indices(manifest, anchors)
    shards = {int(s["id"]): s for s in manifest["shards"]}
    parts = []
    for sid in sorted(grouped):
        rec = shards[sid]
        loc, _ = grouped[sid]
        codes = np.load(rec["codes"], mmap_mode="r")
        parts.append(tq.dequantize(np.ascontiguousarray(codes[loc])))
    return np.concatenate(parts, axis=0)


def scan_codes_for_targets(manifest, tq, targets, chunk_mols, use_cuda):
    """targets: list of dicts with keys q_rot (Q,d), medians, mads, heap.

    q_rot is (n_pock, d) float32 in the *rotated* frame.
    """
    try:
        import torch as _torch
        device = _torch.device(
            "cuda" if use_cuda and _torch.cuda.is_available() else "cpu"
        )
        codebook = _torch.from_numpy(np.ascontiguousarray(tq.codebook)).to(device)
        q_rots = [
            _torch.from_numpy(np.ascontiguousarray(t["q_rot"])).to(device)
            for t in targets
        ]
        use_th = True
    except Exception:
        use_th = False
        device = None
        codebook = None
        q_rots = None

    for rec in tqdm(manifest["shards"], desc="scan shards"):
        codes = np.load(rec["codes"], mmap_mode="r")
        n = int(codes.shape[0])
        off = int(rec["offset"])
        for start in range(0, n, int(chunk_mols)):
            end = min(n, start + int(chunk_mols))
            packed = np.ascontiguousarray(codes[start:end])
            if use_th:
                y = _dequant_chunk_torch(packed, codebook, device, tq=tq)
                local = np.arange(off + start, off + end, dtype=np.int64)
                for t, q in zip(targets, q_rots):
                    scores = (q @ y.t()).detach().float().cpu().numpy()
                    metric = _batch_gate_metric(scores, t["medians"], t["mads"])
                    t["heap"].add_batch(metric, local)
            else:
                y = tq.dequantize(packed)
                local = np.arange(off + start, off + end, dtype=np.int64)
                for t in targets:
                    scores = t["q_rot"] @ y.T
                    metric = _batch_gate_metric(scores, t["medians"], t["mads"])
                    t["heap"].add_batch(metric, local)
    for t in targets:
        idx, metric = t["heap"].finalize()
        t["global_idx"] = idx
        t["metric"] = metric


def load_gate_pocket_reps(task, model, pocket_path, fold_version, gate_fold, use_cuda):
    pocket_dataset = task.load_pockets_dataset(pocket_path)
    pocket_data = task._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)
    ckpts = task._fold_checkpoints(fold_version)
    reps = task._cache_pocket_reps_by_fold(
        model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, True
    )
    return reps[int(gate_fold)]


def run_scan(args, task, model, use_cuda, use_fp16):
    man = load_manifest(args.manifest)
    tq = TurboQuant.load(args.turboquant_path or man["turboquant"])
    if tq.version != man["shards"][0].get("turboquant_version", tq.version):
        logger.warning("turboquant version mismatch vs first shard sidecar")
    topk = int(args.topk)
    chunk = int(args.chunk_mols)
    pocket_paths = [p for p in args.pocket_path.split(",") if p.strip()]
    if not pocket_paths:
        raise ValueError("need at least one --pocket-path")

    y_anchor = gather_anchor_codes(man, tq)
    logger.info(f"anchor sample: {y_anchor.shape[0]} molecules")

    targets = []
    for ppath in pocket_paths:
        ppath = ppath.strip()
        pocket_reps = load_gate_pocket_reps(
            task,
            model,
            ppath,
            man.get("fold_version", args.fold_version),
            int(man.get("gate_fold", args.gate_fold)),
            use_cuda,
        )
        q_rot = tq.rotate_query(pocket_reps)
        scores_a = q_rot @ y_anchor.T
        medians, mads = _robust_pocket_anchors(scores_a)
        targets.append(
            {
                "path": ppath,
                "stem": _target_stem(ppath),
                "q_rot": q_rot,
                "medians": medians,
                "mads": mads,
                "heap": _RunningTopK(topk),
                "n_pock": int(pocket_reps.shape[0]),
            }
        )
        logger.info(
            f"target {ppath}: {pocket_reps.shape[0]} pockets, "
            f"MAD mean={float(np.mean(mads)):.5f}"
        )

    scan_codes_for_targets(man, tq, targets, chunk, use_cuda)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for t in targets:
        path = os.path.join(out_dir, f"{t['stem']}.shortlist.npz")
        write_shortlist(path, t["global_idx"], t["metric"], man)
        logger.info(f"wrote {path} k={t['global_idx'].size}")
        written.append(path)
    return written, man, tq, targets


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
    run_scan(args, task, model, use_cuda, use_fp16)


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--turboquant-path", type=str, default="")
    parser.add_argument(
        "--pocket-path",
        type=str,
        required=True,
        help="comma-separated pocket LMDB paths (one target each)",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=2_000_000)
    parser.add_argument("--chunk-mols", type=int, default=524288)
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--gate-fold", type=int, default=4)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
