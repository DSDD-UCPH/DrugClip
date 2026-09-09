#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Score an LMDB with one fold (full ranking), plus the quantized copy of that fold.

Encodes the requested fold exactly (same fused encode + pocket GEMM as full
retrieval, but a single fold), caches those embeddings, writes those scores,
stores TurboQuant codes for the same embeddings, scores the reconstructed
vectors the same way, writes both full ranked outputs, and reports timing
plus MAE / Pearson / top-frac overlap.
"""

import logging
import gc
import os
import sys
import time

import numpy as np
import torch
from unicore import distributed_utils, options
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.compare_fold_quant")


def _set_stats(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    if not a and not b:
        return 1.0, 1.0, 0
    inter = len(a & b)
    union = len(a | b)
    jaccard = 1.0 if union == 0 else inter / union
    overlap = 1.0 if not a else inter / len(a)
    return jaccard, overlap, inter


def _topk(metric, k):
    k = max(1, min(int(k), int(metric.size)))
    return np.argpartition(metric, -k)[-k:]


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _mae_pearson_chunked(a, b, chunk_mols=16384):
    """MAE and Pearson over a (n_pock, n_mols) pair without a full abs/diff temp."""
    n_pock, n_mols = a.shape
    n = float(n_pock) * float(n_mols)
    if n < 1:
        return 0.0, float("nan")
    abs_sum = 0.0
    sum_a = 0.0
    sum_b = 0.0
    sum_aa = 0.0
    sum_bb = 0.0
    sum_ab = 0.0
    chunk = max(1, int(chunk_mols))
    n_chunks = (int(n_mols) + chunk - 1) // chunk
    logger.info(
        f"chunked MAE/Pearson: {n_pock}x{n_mols} scores, "
        f"{n_chunks} chunks of {chunk} mols"
    )
    t0 = time.perf_counter()
    for i, start in enumerate(range(0, int(n_mols), chunk)):
        end = min(int(n_mols), start + chunk)
        aa = np.asarray(a[:, start:end], dtype=np.float32)
        bb = np.asarray(b[:, start:end], dtype=np.float32)
        d = aa - bb
        abs_sum += float(np.abs(d).sum(dtype=np.float64))
        sum_a += float(aa.sum(dtype=np.float64))
        sum_b += float(bb.sum(dtype=np.float64))
        sum_aa += float(np.square(aa, dtype=np.float64).sum())
        sum_bb += float(np.square(bb, dtype=np.float64).sum())
        sum_ab += float(np.multiply(aa, bb, dtype=np.float64).sum())
        if (i + 1) % max(1, n_chunks // 10) == 0 or (i + 1) == n_chunks:
            logger.info(
                f"chunked MAE/Pearson: {i + 1}/{n_chunks} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    mae = float(abs_sum / n)
    mean_a = sum_a / n
    mean_b = sum_b / n
    cov = sum_ab / n - mean_a * mean_b
    var_a = sum_aa / n - mean_a * mean_a
    var_b = sum_bb / n - mean_b * mean_b
    if var_a <= 0.0 or var_b <= 0.0:
        return mae, float("nan")
    return mae, float(cov / np.sqrt(var_a * var_b))


def _topk_per_pocket(scores, k, label):
    n_pock, n_mols = scores.shape
    k = max(1, min(int(k), int(n_mols)))
    out = np.empty((n_pock, k), dtype=np.int64)
    logger.info(f"argpartition top-{k} {label}: {n_pock} pockets, n_mols={n_mols}")
    t0 = time.perf_counter()
    for p in range(n_pock):
        row = np.asarray(scores[p], dtype=np.float32)
        out[p] = np.argpartition(row, -k)[-k:]
        if (p + 1) % 50 == 0 or p + 1 == n_pock:
            logger.info(
                f"argpartition top-{k} {label}: {p + 1}/{n_pock} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    return out


def _ranks(metric):
    order = np.argsort(-np.asarray(metric).reshape(-1), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    return ranks


def _cuda_sync(use_cuda):
    if use_cuda:
        torch.cuda.synchronize()


def score_pockets(pocket_reps, mol_embs):
    p = np.ascontiguousarray(pocket_reps, dtype=np.float32)
    m = np.ascontiguousarray(mol_embs, dtype=np.float32)
    return p @ m.T


def rank_metric(scores, fold_version):
    from unimol.tasks._drugclip_rank import _max_zscore_metric, _robust_pocket_anchors

    if str(fold_version).startswith("6_folds"):
        medians, mads = _robust_pocket_anchors(scores)
        return _max_zscore_metric(scores, medians, mads), medians, mads
    return np.max(np.asarray(scores, dtype=np.float32), axis=0), None, None


def write_ranked(path, names, metric):
    metric = np.asarray(metric).reshape(-1)
    order = np.argsort(-metric, kind="mergesort")
    with open(path, "w") as f:
        for i in order:
            name = names[i] if names[i] is not None else str(int(i))
            f.write(f"{name},{metric[i]}\n")
    logger.info(f"wrote {len(order)} ranked rows to {path}")


def _npy_rows(path, n_mols):
    if not os.path.isfile(path):
        return None
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return None
    if arr.ndim != 2 or int(arr.shape[0]) != int(n_mols):
        return None
    return arr


def _load_names(path, n_mols):
    if not os.path.isfile(path):
        return None
    names = list(np.load(path, allow_pickle=True))
    if len(names) != int(n_mols):
        return None
    return [None if n is None else str(n) for n in names]


def _atomic_npy(path, arr):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_cached_fold_embs(out_dir, task, fold_version, gate_fold, n_mols):
    """Return (embs float32, names) or (None, None) if no usable cache."""
    local_emb = os.path.join(out_dir, f"fold{gate_fold}_embs.npy")
    local_names = os.path.join(out_dir, "names.npy")
    arr = _npy_rows(local_emb, n_mols)
    if arr is not None:
        names = _load_names(local_names, n_mols)
        if names is not None:
            logger.info(
                f"loaded fold-{gate_fold} embeddings from {local_emb} "
                f"shape={arr.shape} dtype={arr.dtype}"
            )
            return np.ascontiguousarray(arr, dtype=np.float32), names
    npy = task._mol_npy_cache_paths(fold_version)
    arr = _npy_rows(npy["folds"][int(gate_fold)], n_mols)
    names = _load_names(npy["names"], n_mols)
    if arr is None or names is None:
        return None, None
    logger.info(
        f"loaded fold-{gate_fold} embeddings from {npy['folds'][int(gate_fold)]} "
        f"shape={arr.shape} dtype={arr.dtype}"
    )
    embs = np.ascontiguousarray(arr, dtype=np.float32)
    return embs, names


def save_fold_embs(
    out_dir, task, fold_version, gate_fold, embs, names, write_global=True
):
    local_emb = os.path.join(out_dir, f"fold{gate_fold}_embs.npy")
    local_names = os.path.join(out_dir, "names.npy")
    if _npy_rows(local_emb, embs.shape[0]) is None:
        _atomic_npy(local_emb, np.ascontiguousarray(embs, dtype=np.float32))
        logger.info(f"wrote fold embeddings {local_emb} shape={embs.shape}")
    if _load_names(local_names, embs.shape[0]) is None:
        np.save(local_names, np.asarray(names, dtype=object))
    if not write_global:
        return
    npy = task._mol_npy_cache_paths(fold_version)
    global_emb = npy["folds"][int(gate_fold)]
    existing = _npy_rows(global_emb, embs.shape[0])
    if existing is None:
        os.makedirs(npy["dir"], exist_ok=True)
        _atomic_npy(global_emb, np.ascontiguousarray(embs, dtype=np.float16))
        logger.info(f"wrote mol npy cache {global_emb} dtype=float16")
    if _load_names(npy["names"], embs.shape[0]) is None:
        os.makedirs(npy["dir"], exist_ok=True)
        np.save(npy["names"], np.asarray(names, dtype=object))
        logger.info(f"wrote mol npy cache names {npy['names']}")


def make_codec(bits, dim, seed, turboquant_path=""):
    from unimol.turboquant import (
        TurboQuant,
        codec_version,
        fit_lloyd_max,
        sample_sphere_coords,
    )

    bits = int(bits)
    if turboquant_path:
        tq = TurboQuant.load(turboquant_path)
        if tq.dim != int(dim):
            raise ValueError(f"turboquant dim {tq.dim} != embedding dim {dim}")
        if tq.bits == bits:
            return tq
        logger.info(
            f"loaded turboquant bits={tq.bits}; refitting {bits}-bit codebook "
            f"on the same rotation"
        )
        samples = sample_sphere_coords(dim, 200000, seed=int(seed) + 1)
        codebook = fit_lloyd_max(samples, 1 << bits, n_iter=80)
        return TurboQuant(
            tq.rotation,
            codebook,
            version=codec_version(dim, bits),
            bits=bits,
        )
    return TurboQuant.create(dim=dim, bits=bits, seed=int(seed))


def compare_outputs(exact_scores, quant_scores, exact_metric, quant_metric, fracs):
    logger.info("MAE/Pearson over pocket-score matrices")
    mae, pearson = _mae_pearson_chunked(exact_scores, quant_scores)
    logger.info(f"pocket-score MAE={mae:.6f} Pearson={pearson:.4f}")
    mae_m = float(np.mean(np.abs(np.asarray(exact_metric) - np.asarray(quant_metric))))
    pearson_m = _pearson(exact_metric, quant_metric)
    spearman = _pearson(_ranks(exact_metric), _ranks(quant_metric))
    logger.info(
        f"rank-metric MAE={mae_m:.6f} Pearson={pearson_m:.4f} Spearman={spearman:.4f}"
    )
    n_mols = int(exact_metric.size)
    frac_list = sorted({float(f) for f in fracs})
    ks = sorted(
        {max(1, int(round(n_mols * float(frac)))) for frac in frac_list}
    )
    exact_pock_top = {}
    quant_pock_top = {}
    exact_met_top = {}
    quant_met_top = {}
    for k in ks:
        exact_pock_top[k] = _topk_per_pocket(exact_scores, k, "exact-pockets")
        quant_pock_top[k] = _topk_per_pocket(quant_scores, k, "quant-pockets")
        exact_met_top[k] = _topk(exact_metric, k)
        quant_met_top[k] = _topk(quant_metric, k)
    rows = []
    for frac in frac_list:
        k = max(1, int(round(n_mols * float(frac))))
        pock_jacs = []
        pock_olaps = []
        for p in range(exact_scores.shape[0]):
            jac, olap, _ = _set_stats(exact_pock_top[k][p], quant_pock_top[k][p])
            pock_jacs.append(jac)
            pock_olaps.append(olap)
        jac_metric, olap_metric, hits_metric = _set_stats(
            exact_met_top[k], quant_met_top[k]
        )
        rows.append(
            {
                "frac": float(frac),
                "k": k,
                "jaccard_pockets": float(np.mean(pock_jacs)),
                "jaccard_metric": float(jac_metric),
                "overlap_pockets": float(np.mean(pock_olaps)),
                "overlap_metric": float(olap_metric),
                "hits_metric": int(hits_metric),
            }
        )
    cross = []
    for exact_frac in frac_list:
        k_exact = max(1, int(round(n_mols * exact_frac)))
        for quant_frac in frac_list:
            if quant_frac <= exact_frac:
                continue
            k_quant = max(1, int(round(n_mols * quant_frac)))
            _, olap_metric, hits_metric = _set_stats(
                exact_met_top[k_exact], quant_met_top[k_quant]
            )
            pock_olaps = [
                _set_stats(exact_pock_top[k_exact][p], quant_pock_top[k_quant][p])[1]
                for p in range(exact_scores.shape[0])
            ]
            cross.append(
                {
                    "exact_frac": exact_frac,
                    "quant_frac": float(quant_frac),
                    "k_exact": k_exact,
                    "k_quant": k_quant,
                    "overlap_pockets": float(np.mean(pock_olaps)),
                    "overlap_metric": float(olap_metric),
                    "hits_metric": int(hits_metric),
                }
            )
    return {
        "mae": mae,
        "pearson": pearson,
        "mae_metric": mae_m,
        "pearson_metric": pearson_m,
        "spearman_metric": spearman,
        "fracs": rows,
        "cross": cross,
    }


def _mols_per_sec(n_mols, elapsed):
    if elapsed is None or elapsed <= 0:
        return 0.0
    return float(n_mols) / float(elapsed)


def format_report(args, n_mols, n_pock, mad_mean, mad_median, stats, timings):
    t_encode = timings["encode"]
    t_exact = timings["exact_score"]
    t_quantize = timings["quantize"]
    t_quant = timings["quant_score"]
    exact_e2e = t_encode + t_exact
    mps_exact = _mols_per_sec(n_mols, t_exact)
    mps_quantize = _mols_per_sec(n_mols, t_quantize)
    mps_quant = _mols_per_sec(n_mols, t_quant)
    score_x = (t_exact / t_quant) if t_quant > 0 else float("nan")
    e2e_x = (exact_e2e / t_quant) if t_quant > 0 else float("nan")
    lines = [
        f"fold-quant compare: n_mols={n_mols} n_pock={n_pock} "
        f"fold={int(args.gate_fold)} bits={int(args.quant_bits)} "
        f"fold_version={args.fold_version}",
        f"exact per-pocket MAD: mean={mad_mean:.5f} median={mad_median:.5f}",
        f"timing encode={t_encode:.3f}s  exact_score={t_exact:.3f}s "
        f"({mps_exact:.1f} mol/s)  quantize={t_quantize:.3f}s "
        f"({mps_quantize:.1f} mol/s)  quant_score={t_quant:.3f}s "
        f"({mps_quant:.1f} mol/s)",
        f"timing speedup score={score_x:.2f}x  "
        f"encode+score vs quant={e2e_x:.2f}x",
        f"pocket-score MAE={stats['mae']:.6f} Pearson={stats['pearson']:.4f}",
        f"rank-metric MAE={stats['mae_metric']:.6f} Pearson={stats['pearson_metric']:.4f} "
        f"Spearman={stats['spearman_metric']:.4f}",
        f"{'frac':>10} | {'k':>8} | {'overlap_pock':>12} | {'overlap_rank':>12} | "
        f"{'hits':>14} | {'Jac_pock':>10} | {'Jac_rank':>10}",
        "-" * 92,
    ]
    for row in stats["fracs"]:
        lines.append(
            f"{row['frac']:10.6f} | {row['k']:8d} | "
            f"{row['overlap_pockets']:12.2%} | {row['overlap_metric']:12.2%} | "
            f"{row['hits_metric']:6d}/{row['k']:<6d} | "
            f"{row['jaccard_pockets']:10.4f} | {row['jaccard_metric']:10.4f}"
        )
    cross = stats.get("cross") or []
    if cross:
        lines.extend(
            [
                "",
                "exact smaller top in quantized larger top "
                "(recall of exact top-exact_frac inside quantized top-quant_frac):",
                f"{'exact_frac':>10} | {'quant_frac':>10} | {'k_exact':>8} | "
                f"{'k_quant':>8} | {'overlap_pock':>12} | {'overlap_rank':>12} | {'hits':>14}",
                "-" * 92,
            ]
        )
        for row in cross:
            lines.append(
                f"{row['exact_frac']:10.6f} | {row['quant_frac']:10.6f} | "
                f"{row['k_exact']:8d} | {row['k_quant']:8d} | "
                f"{row['overlap_pockets']:12.2%} | {row['overlap_metric']:12.2%} | "
                f"{row['hits_metric']:6d}/{row['k_exact']:<6d}"
            )
    return "\n".join(lines)


def main(args):
    from unimol.index_encode import encode_fold_embeddings, load_fold_encoder
    from unimol.tasks.drugclip import RETRIEVAL_TOP_FRAC

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    bits = int(args.quant_bits)
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"--quant-bits must be 1, 2, 3, or 4, got {bits}")
    gate_fold = int(args.gate_fold)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    task = tasks.setup_task(args)
    model = task.build_model(args)
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()
    model.eval()

    bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else 256

    pocket_dataset = task.load_pockets_dataset(args.pocket_path)
    pocket_data = task._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)
    ckpts = task._fold_checkpoints(args.fold_version)
    pocket_reps = task._cache_pocket_reps_by_fold(
        model, ckpts, pocket_data, args.pocket_path, args.fold_version, use_cuda, True
    )[gate_fold]

    mol_model, mol_project = None, None
    mol_dataset = task.load_mols_dataset(
        args.mol_path, "atoms", "coordinates", readahead=True
    )
    n_full = len(mol_dataset)
    if args.max_mols and args.max_mols > 0:
        n = min(int(args.max_mols), n_full)
        mol_dataset = task._subset_mol_dataset(mol_dataset, np.arange(n))
    n_mols = len(mol_dataset)
    write_global = n_mols == n_full

    if use_cuda:
        _ = torch.zeros((1, 1), device="cuda") @ torch.zeros((1, 1), device="cuda")
        _cuda_sync(True)

    embs, names = load_cached_fold_embs(
        out_dir, task, args.fold_version, gate_fold, n_mols
    )
    if embs is not None:
        t_encode = 0.0
        logger.info(f"skipped encode; using cached fold-{gate_fold} embeddings")
        save_fold_embs(
            out_dir,
            task,
            args.fold_version,
            gate_fold,
            embs,
            names,
            write_global=write_global,
        )
    else:
        mol_model, mol_project = load_fold_encoder(
            task, model, gate_fold, args.fold_version, use_cuda, use_fp16
        )
        _cuda_sync(use_cuda)
        t0 = time.perf_counter()
        embs, names = encode_fold_embeddings(
            task,
            mol_model,
            mol_project,
            mol_dataset,
            use_cuda,
            bsz,
            use_length_bucket=bool(args.length_bucket),
            run_label=f"fold {gate_fold} embeddings",
            collect_names=True,
        )
        _cuda_sync(use_cuda)
        t_encode = time.perf_counter() - t0
        save_fold_embs(
            out_dir,
            task,
            args.fold_version,
            gate_fold,
            embs,
            names,
            write_global=write_global,
        )
        del mol_model, mol_project
        if use_cuda:
            torch.cuda.empty_cache()

    _cuda_sync(use_cuda)
    t0 = time.perf_counter()
    exact_scores = score_pockets(pocket_reps, embs)
    _cuda_sync(use_cuda)
    t_exact_score = time.perf_counter() - t0
    exact_metric, medians, mads = rank_metric(exact_scores, args.fold_version)
    if mads is not None:
        mad_mean = float(np.mean(mads))
        mad_median = float(np.median(mads))
    else:
        mad_mean = mad_median = float("nan")
    logger.info(
        f"exact fold-{gate_fold} scores: shape={exact_scores.shape} "
        f"MAD mean={mad_mean:.5f} median={mad_median:.5f}"
    )

    tq = make_codec(bits, embs.shape[1], args.seed, args.turboquant_path or "")
    tq_path = os.path.join(out_dir, "turboquant.npz")
    tq.save(tq_path)
    logger.info(f"wrote {tq_path} bits={tq.bits} version={tq.version}")

    from unimol.turboquant import (
        dequant_gemm_numpy,
        dequant_gemm_torch,
        write_codes_chunked,
    )

    codes_path = os.path.join(out_dir, f"fold{gate_fold}.codes.npy")
    logger.info(
        f"chunked quantize fold-{gate_fold} embeddings "
        f"shape={embs.shape} bits={tq.bits} -> {codes_path}"
    )
    t0 = time.perf_counter()
    write_codes_chunked(codes_path, embs, tq)
    t_quantize = time.perf_counter() - t0
    logger.info(
        f"wrote quantized fold codes {codes_path} "
        f"quantize={t_quantize:.3f}s ({_mols_per_sec(embs.shape[0], t_quantize):.1f} mol/s)"
    )
    del embs
    gc.collect()
    codes = np.load(codes_path, mmap_mode="r")

    _cuda_sync(use_cuda)
    t0 = time.perf_counter()
    if getattr(tq, "packed", tq.bits == 4):
        quant_gemm = dequant_gemm_torch if use_cuda else dequant_gemm_numpy
        quant_scores = quant_gemm(pocket_reps, codes, tq)
    else:
        p = np.ascontiguousarray(pocket_reps, dtype=np.float32)
        n = int(codes.shape[0])
        quant_scores = np.empty((p.shape[0], n), dtype=np.float32)
        chunk = 65536
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            hat = tq.reconstruct(codes[start:end])
            quant_scores[:, start:end] = p @ hat.T
    _cuda_sync(use_cuda)
    t_quant_score = time.perf_counter() - t0
    quant_metric, _, _ = rank_metric(quant_scores, args.fold_version)

    exact_scores_path = os.path.join(out_dir, "exact_scores.npy")
    quant_scores_path = os.path.join(out_dir, "quant_scores.npy")
    np.save(exact_scores_path, exact_scores)
    np.save(quant_scores_path, quant_scores)
    np.save(os.path.join(out_dir, "exact_metric.npy"), exact_metric)
    np.save(os.path.join(out_dir, "quant_metric.npy"), quant_metric)
    write_ranked(os.path.join(out_dir, "exact_full.txt"), names, exact_metric)
    write_ranked(os.path.join(out_dir, "quant_full.txt"), names, quant_metric)

    fracs = [float(x) for x in str(args.top_fracs).split(",") if x.strip()]
    if RETRIEVAL_TOP_FRAC not in fracs:
        fracs.append(float(RETRIEVAL_TOP_FRAC))
    logger.info("comparing exact vs quantized scores (chunked MAE/Pearson)")
    stats = compare_outputs(
        exact_scores, quant_scores, exact_metric, quant_metric, fracs
    )
    n_mols = exact_scores.shape[1]
    n_pock = exact_scores.shape[0]
    timings = {
        "encode": t_encode,
        "exact_score": t_exact_score,
        "quantize": t_quantize,
        "quant_score": t_quant_score,
    }
    logger.info(
        f"timing encode={t_encode:.3f}s exact_score={t_exact_score:.3f}s "
        f"({_mols_per_sec(n_mols, t_exact_score):.1f} mol/s) "
        f"quantize={t_quantize:.3f}s "
        f"({_mols_per_sec(n_mols, t_quantize):.1f} mol/s) "
        f"quant_score={t_quant_score:.3f}s "
        f"({_mols_per_sec(n_mols, t_quant_score):.1f} mol/s)"
    )
    report = format_report(
        args, n_mols, n_pock, mad_mean, mad_median, stats, timings
    )
    print(report)
    report_path = args.report_path or os.path.join(out_dir, "compare.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    logger.info(f"wrote {report_path}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, required=True)
    parser.add_argument("--pocket-path", type=str, required=True)
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--gate-fold", type=int, default=4)
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="TurboQuant bit width (1-4). Codes are packed (1/2/3/4-bit).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./fold_quant_compare",
        help="directory for full exact/quant score dumps, codes, and the report",
    )
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--max-mols", type=int, default=0)
    parser.add_argument("--length-bucket", type=int, default=1)
    parser.add_argument("--turboquant-path", type=str, default="")
    parser.add_argument(
        "--top-fracs",
        type=str,
        default="0.0001,0.001,0.01",
        help="comma-separated top fractions for overlap / Jaccard (ranking metric and per-pocket)",
    )
    parser.add_argument("--report-path", type=str, default="")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
