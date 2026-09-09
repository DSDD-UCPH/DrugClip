#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 0: can fold 4 alone gate at production tightness?

Scores one shard through every fold (exact fp16, no quantization), ranks with
the same mean / median-MAD / max-over-pockets path as retrieval, and reports
recall of the full-ensemble top-N inside the fold-4 top-(mult*N).

Default N/n_mols = 1e-4 (production shortlist tightness on a 100B library);
existing cascade validation only covers cascade_frac=0.2 (~2000x looser).
"""

import logging
import os
import sys

import numpy as np
import torch
from unicore import distributed_utils, options
from unicore import tasks

from unimol.index_encode import load_all_fold_encoders, score_folds_streaming
from unimol.tasks._drugclip_rank import _max_zscore_metric, _robust_pocket_anchors
from unimol.tasks.drugclip import RETRIEVAL_TOP_FRAC

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.fold_gate")


def _parse_fracs(raw):
    return tuple(float(x) for x in str(raw).split(",") if x.strip())


def _topk_indices(metric, k):
    k = max(1, min(int(k), int(metric.size)))
    sel = np.argpartition(metric, -k)[-k:]
    return sel


def report_fold_gate(
    mean_scores,
    gate_scores,
    fold_version,
    n_fracs,
    gate_mult,
    save_path=None,
):
    n_pock, n_mols = mean_scores.shape
    if fold_version.startswith("6_folds"):
        medians, mads = _robust_pocket_anchors(mean_scores)
        ens = _max_zscore_metric(mean_scores, medians, mads)
        g_med, g_mad = _robust_pocket_anchors(gate_scores)
        gate = _max_zscore_metric(gate_scores, g_med, g_mad)
        mad_mean = float(np.mean(g_mad))
        mad_median = float(np.median(g_mad))
    else:
        ens = np.max(mean_scores, axis=0)
        gate = np.max(gate_scores, axis=0)
        mad_mean = mad_median = float("nan")

    lines = [
        f"fold-gate report: {n_mols} mols, {n_pock} pocket(s), "
        f"gate_mult={gate_mult}, ensemble top-frac baseline={RETRIEVAL_TOP_FRAC}",
        f"gate-fold per-pocket MAD: mean={mad_mean:.5f} median={mad_median:.5f}",
        f"{'n_frac':>10} | {'N':>8} | {'gate_k':>8} | {'recall':>8} | hits",
        "-" * 52,
    ]
    for n_frac in n_fracs:
        n_true = max(1, int(round(n_mols * float(n_frac))))
        n_gate = max(n_true, int(round(n_true * float(gate_mult))))
        n_gate = min(n_gate, n_mols)
        true_set = set(_topk_indices(ens, n_true).tolist())
        gate_set = set(_topk_indices(gate, n_gate).tolist())
        hit = len(true_set & gate_set)
        recall = hit / len(true_set) if true_set else float("nan")
        lines.append(
            f"{n_frac:10.6f} | {n_true:8d} | {n_gate:8d} | {recall:8.2%} | "
            f"{hit}/{len(true_set)}"
        )
        logger.info(
            f"n_frac={n_frac} N={n_true} gate_k={n_gate} recall={recall:.4f}"
        )
    report = "\n".join(lines)
    print(report)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report + "\n")
    return report


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

    bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else 256
    gate_fold = int(args.gate_fold)
    n_fracs = _parse_fracs(args.n_fracs)

    pocket_dataset = task.load_pockets_dataset(args.pocket_path)
    pocket_data = task._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)
    ckpts = task._fold_checkpoints(args.fold_version)
    pocket_reps = task._cache_pocket_reps_by_fold(
        model, ckpts, pocket_data, args.pocket_path, args.fold_version, use_cuda, True
    )
    encoders = load_all_fold_encoders(
        task, model, args.fold_version, use_cuda, use_fp16
    )
    mol_dataset = task.load_mols_dataset(
        args.mol_path, "atoms", "coordinates", readahead=True
    )
    if args.max_mols and args.max_mols > 0:
        n = min(int(args.max_mols), len(mol_dataset))
        mol_dataset = task._subset_mol_dataset(mol_dataset, np.arange(n))

    mean_scores, gate_scores, _ = score_folds_streaming(
        task,
        encoders,
        pocket_reps,
        mol_dataset,
        use_cuda,
        bsz,
        gate_fold=gate_fold,
        use_length_bucket=bool(args.length_bucket),
        run_label=f"fold-gate (gate={gate_fold})",
    )
    report_fold_gate(
        mean_scores,
        gate_scores,
        args.fold_version,
        n_fracs,
        args.gate_mult,
        save_path=args.report_path or None,
    )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, required=True)
    parser.add_argument("--pocket-path", type=str, required=True)
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--gate-fold", type=int, default=4)
    parser.add_argument(
        "--n-fracs",
        type=str,
        default="0.0001,0.0005,0.001,0.01",
        help="comma-separated true top-N fractions of this shard",
    )
    parser.add_argument(
        "--gate-mult",
        type=float,
        default=10.0,
        help="keep this times N from the gate fold",
    )
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--max-mols", type=int, default=0)
    parser.add_argument("--length-bucket", type=int, default=1)
    parser.add_argument("--report-path", type=str, default="")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
