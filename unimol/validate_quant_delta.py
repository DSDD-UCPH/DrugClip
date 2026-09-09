#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 0: quantization delta vs exact fold-4 fp16.

Encodes fold 4 exactly, then compares 4-bit MSE-only TurboQuant (and 3-bit /
prod-variant controls) against those embeddings on pocket scores: top-1%
Jaccard, MAE, Pearson, plus the measured per-pocket MAD.
"""

import logging
import os
import sys

import numpy as np
import torch
from unicore import distributed_utils, options
from unicore import tasks

from unimol.index_encode import encode_fold_embeddings, load_fold_encoder
from unimol.turboquant import (
    TurboQuant,
    fit_lloyd_max,
    mse_reconstruct_unpacked,
    quantize_prod_control,
    sample_sphere_coords,
)
from unimol.tasks._drugclip_rank import _robust_pocket_anchors

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.quant_delta")


def _jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _topk(metric, k):
    k = max(1, min(int(k), int(metric.size)))
    return np.argpartition(metric, -k)[-k:]


def _pearson(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_pockets(pocket_reps, mol_embs):
    p = np.ascontiguousarray(pocket_reps, dtype=np.float32)
    m = np.ascontiguousarray(mol_embs, dtype=np.float32)
    return p @ m.T


def compare_arm(name, exact_scores, hat_embs, pocket_reps, top_frac):
    hat_scores = score_pockets(pocket_reps, hat_embs)
    mae = float(np.mean(np.abs(exact_scores - hat_scores)))
    pearson = _pearson(exact_scores, hat_scores)
    n_mols = exact_scores.shape[1]
    k = max(1, int(round(n_mols * float(top_frac))))
    jacs = []
    for p in range(exact_scores.shape[0]):
        jacs.append(_jaccard(_topk(exact_scores[p], k), _topk(hat_scores[p], k)))
    # Max-over-pockets lists (no z-score): ranking overlap of the fused metric.
    exact_max = np.max(exact_scores, axis=0)
    hat_max = np.max(hat_scores, axis=0)
    jac_max = _jaccard(_topk(exact_max, k), _topk(hat_max, k))
    return {
        "name": name,
        "mae": mae,
        "pearson": pearson,
        "jaccard_topfrac_mean_over_pockets": float(np.mean(jacs)),
        "jaccard_topfrac_max_over_pockets": float(jac_max),
        "k": k,
    }


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

    pocket_dataset = task.load_pockets_dataset(args.pocket_path)
    pocket_data = task._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)
    ckpts = task._fold_checkpoints(args.fold_version)
    pocket_reps = task._cache_pocket_reps_by_fold(
        model, ckpts, pocket_data, args.pocket_path, args.fold_version, use_cuda, True
    )[gate_fold]

    mol_model, mol_project = load_fold_encoder(
        task, model, gate_fold, args.fold_version, use_cuda, use_fp16
    )
    mol_dataset = task.load_mols_dataset(
        args.mol_path, "atoms", "coordinates", readahead=True
    )
    if args.max_mols and args.max_mols > 0:
        n = min(int(args.max_mols), len(mol_dataset))
        mol_dataset = task._subset_mol_dataset(mol_dataset, np.arange(n))

    embs = encode_fold_embeddings(
        task,
        mol_model,
        mol_project,
        mol_dataset,
        use_cuda,
        bsz,
        use_length_bucket=bool(args.length_bucket),
        run_label=f"fold {gate_fold} embeddings",
    )
    exact_scores = score_pockets(pocket_reps, embs)
    medians, mads = _robust_pocket_anchors(exact_scores)
    mad_mean = float(np.mean(mads))
    mad_median = float(np.median(mads))
    logger.info(
        f"exact fold-{gate_fold} scores: shape={exact_scores.shape} "
        f"MAD mean={mad_mean:.5f} median={mad_median:.5f}"
    )

    if args.turboquant_path:
        tq4 = TurboQuant.load(args.turboquant_path)
    else:
        tq4 = TurboQuant.create(
            dim=embs.shape[1], bits=4, seed=int(args.seed), n_samples=200000
        )
        if args.save_turboquant:
            tq4.save(args.save_turboquant)
            logger.info(f"wrote {args.save_turboquant}")

    recon4 = tq4.reconstruct(tq4.quantize(embs))
    samples = sample_sphere_coords(embs.shape[1], 200000, seed=int(args.seed) + 1)
    cb3 = fit_lloyd_max(samples, 8, n_iter=40)
    recon3, _ = mse_reconstruct_unpacked(embs, tq4.rotation, cb3)

    class _Low:
        rotation = tq4.rotation
        codebook = cb3

        def rotate(self, x):
            return x @ tq4.rotation

        def quantize_indices(self, y, already_rotated=False):
            yy = y if already_rotated else self.rotate(y)
            dist = np.abs(yy[..., None] - cb3.reshape(1, 1, -1))
            return dist.argmin(axis=-1).astype(np.uint8)

    recon_prod = quantize_prod_control(embs, _Low(), bits=4, seed=int(args.seed))

    arms = [
        compare_arm("mse-4bit", exact_scores, recon4, pocket_reps, args.top_frac),
        compare_arm("mse-3bit", exact_scores, recon3, pocket_reps, args.top_frac),
        compare_arm("prod-4bit-control", exact_scores, recon_prod, pocket_reps, args.top_frac),
    ]
    lines = [
        f"quant-delta report: n_mols={embs.shape[0]} n_pock={exact_scores.shape[0]} "
        f"top_frac={args.top_frac} gate_fold={gate_fold}",
        f"exact per-pocket MAD: mean={mad_mean:.5f} median={mad_median:.5f} "
        f"(analytic 4-bit RMSE~0.0084; compare against MAD)",
        f"{'arm':<20} {'MAE':>10} {'Pearson':>10} {'Jac_pock':>10} {'Jac_max':>10} k",
        "-" * 72,
    ]
    for a in arms:
        lines.append(
            f"{a['name']:<20} {a['mae']:10.5f} {a['pearson']:10.4f} "
            f"{a['jaccard_topfrac_mean_over_pockets']:10.4f} "
            f"{a['jaccard_topfrac_max_over_pockets']:10.4f} {a['k']}"
        )
    report = "\n".join(lines)
    print(report)
    if args.report_path:
        with open(args.report_path, "w") as f:
            f.write(report + "\n")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, required=True)
    parser.add_argument("--pocket-path", type=str, required=True)
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--gate-fold", type=int, default=4)
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--max-mols", type=int, default=50000)
    parser.add_argument("--top-frac", type=float, default=0.01)
    parser.add_argument("--length-bucket", type=int, default=1)
    parser.add_argument("--turboquant-path", type=str, default="")
    parser.add_argument("--save-turboquant", type=str, default="")
    parser.add_argument("--report-path", type=str, default="")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
