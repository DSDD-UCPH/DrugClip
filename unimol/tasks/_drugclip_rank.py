# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Cascade / retrieval ranking helpers for DrugCLIP (not a Unicore task)."""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

_SCORE_MEMMAP_SELECT_CHUNK = 1_000_000
_DEFAULT_PICKLE_SCORE_CHUNK = 262144


def _pickle_score_chunk_size():
    # Mol-chunk width for cached embedding GEMM (npy mmap / legacy pickle).
    raw = os.environ.get("DRUGCLIP_PICKLE_SCORE_CHUNK", "").strip()
    if raw:
        try:
            return max(4096, int(raw))
        except ValueError:
            pass
    return _DEFAULT_PICKLE_SCORE_CHUNK


def _env_float_dtype(name, default="float16"):
    # Parse DRUGCLIP_* dtype env vars: float16 by default, float32 on request.
    env = os.environ.get(name, default).lower()
    if env in ("float32", "fp32", "f32"):
        return np.float32
    return np.float16


def _score_memmap_dtype():
    # On-disk fold-mean score dtype. Default fp16 (~2x smaller than fp32); set
    # DRUGCLIP_SCORE_MEMMAP_DTYPE=float32 for bit-exact ranking vs in-memory path.
    return _env_float_dtype("DRUGCLIP_SCORE_MEMMAP_DTYPE")


def _cascade_score_memmap_dtype():
    # Tier-0 cascade scratch memmap dtype. Default fp16 (~2x less I/O for
    # select_gather); set DRUGCLIP_CASCADE_SCORE_DTYPE=float32 for the old path.
    return _env_float_dtype("DRUGCLIP_CASCADE_SCORE_DTYPE")


def _cascade_anchor_sample_size():
    # Per-pocket subsample size for encode-time anchors. 0 disables the reservoir
    # and falls back to a full memmap/dense anchor pass.
    try:
        return max(0, int(os.environ.get("DRUGCLIP_CASCADE_ANCHOR_SAMPLE", "262144")))
    except ValueError:
        return 262144


def _cascade_anchor_exact():
    # Force partition-based exact median/MAD (skip histogram approximation).
    return os.environ.get("DRUGCLIP_CASCADE_ANCHOR_EXACT", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _cascade_anchor_hist_bins():
    try:
        return max(16, int(os.environ.get("DRUGCLIP_CASCADE_ANCHOR_HIST_BINS", "8192")))
    except ValueError:
        return 8192


def _ceil_to_batch(n, bsz):
    # Round n up to the next multiple of bsz (at least one batch).
    n = max(1, int(n))
    return ((n + bsz - 1) // bsz) * bsz


def _cascade_tier_bsz(base_bsz, n_folds, n_active_folds):
    # Scale DataLoader batch size so n_active_folds * tier_bsz matches the
    # full-ensemble work assumed when base_bsz was tuned (n_folds encoders per batch).
    if n_active_folds == 4:
        return 512
    else:
        return max(1, base_bsz * n_folds // n_active_folds)


def _cascade_score_bytes(n_pock, n_mols, dtype=np.float32):
    return int(n_pock) * int(n_mols) * int(np.dtype(dtype).itemsize)


def _cascade_scores_need_memmap(n_pock, n_mols):
    # Stream scores to a scratch memmap when a dense (n_pock, N) float32 matrix
    # would exceed 2 GiB (e.g. 400 pockets x ~12M mols ≈ 19 GB).
    return _cascade_score_bytes(n_pock, n_mols, np.float32) > (
        2 * 1024 * 1024 * 1024
    )


def _mean_fold_scores(fold_scores, consume=False):
    # Mean over folds into a single (n_pock, n_mols) float32 matrix.
    # Avoids np.stack(...).mean, which materializes a (n_folds, n_pock, n_mols)
    # temporary (tens of GiB for hundreds of pockets x multi-M mols).
    #
    # When consume=True, fold_scores must be a mutable sequence: each entry is
    # set to None after it is added so peak RAM stays near one matrix.
    if consume:
        n = len(fold_scores)
        if n == 0:
            raise ValueError("fold_scores must be non-empty")
        res = np.asarray(fold_scores[0], dtype=np.float32).copy()
        fold_scores[0] = None
        for i in range(1, n):
            res += np.asarray(fold_scores[i], dtype=np.float32)
            fold_scores[i] = None
        if n > 1:
            res /= np.float32(n)
        return res
    fold_list = list(fold_scores)
    if not fold_list:
        raise ValueError("fold_scores must be non-empty")
    res = np.asarray(fold_list[0], dtype=np.float32).copy()
    for fs in fold_list[1:]:
        res += np.asarray(fs, dtype=np.float32)
    if len(fold_list) > 1:
        res /= np.float32(len(fold_list))
    return res


def _ensemble_rank_mols(fold_scores, fold_version, consume=False, return_timings=False):
    # Reference ranking shared by full mode and the cascade final rescore: mean
    # over folds, robust z-score per pocket (6_folds variants), then max over
    # pockets. `fold_scores` is a sequence of (n_pock, n_mols) arrays, one per
    # fold; returns a (n_mols,) score vector.
    #
    # Accumulate the fold mean in-place instead of np.stack(...).mean — stacking
    # 6 x (n_pock, n_pool) float32 matrices (e.g. 400 x 2.3M) allocates tens of
    # GiB and was pushing the cascade final rank into multi-minute swap.
    # Pass consume=True when the caller can release each fold matrix after it is
    # added (cascade final rank); default False preserves other callers.
    t0 = time.perf_counter()
    res = _mean_fold_scores(fold_scores, consume=consume)
    t_mean = time.perf_counter() - t0
    timings = {"mean": t_mean, "anchors": 0.0, "zscore": 0.0}
    if fold_version.startswith("6_folds"):
        t1 = time.perf_counter()
        medians, mads = _robust_pocket_anchors(res)
        timings["anchors"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        # Dense pool: use a larger working-set chunk than the memmap default so
        # we are not Python-loop bound on hundreds of tiny slices.
        res = _max_zscore_metric(
            res, medians, mads, chunk_size=_dense_score_chunk_size(res.shape[0])
        )
        timings["zscore"] = time.perf_counter() - t2
        if return_timings:
            return res, timings
        return res
    t3 = time.perf_counter()
    out = np.max(res, axis=0)
    timings["zscore"] = time.perf_counter() - t3
    if return_timings:
        return out, timings
    return out


def _pocket_median_mad(row):
    # Exact median + MAD via np.partition (faster than two full np.median sorts).
    row = np.asarray(row, dtype=np.float32).ravel()
    n = int(row.size)
    if n == 0:
        return np.float32(0), np.float32(0)
    if n == 1:
        return np.float32(row[0]), np.float32(0)
    mid = n // 2
    a = row.copy()
    if n % 2 == 1:
        med = np.float32(np.partition(a, mid)[mid])
    else:
        part = np.partition(a, [mid - 1, mid])
        med = np.float32(0.5 * (part[mid - 1] + part[mid]))
    abs_dev = np.abs(row - med)
    if n % 2 == 1:
        mad = np.float32(np.partition(abs_dev, mid)[mid])
    else:
        part = np.partition(abs_dev, [mid - 1, mid])
        mad = np.float32(0.5 * (part[mid - 1] + part[mid]))
    return med, mad


def _pocket_median_mad_histogram(row, n_bins=8192):
    # Approximate median + MAD from fixed-bin histograms (O(N), no full sort).
    # Deterministic for a given row; small bias vs exact median is acceptable for
    # cascade *gating* anchors (reservoir / memmap fallback only — not final rank).
    row = np.asarray(row, dtype=np.float32).ravel()
    n = int(row.size)
    if n == 0:
        return np.float32(0), np.float32(0)
    if n == 1:
        return np.float32(row[0]), np.float32(0)
    lo = float(np.min(row))
    hi = float(np.max(row))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.float32(lo if np.isfinite(lo) else 0.0)
        return med, np.float32(0)
    n_bins = max(16, int(n_bins))
    counts, edges = np.histogram(row, bins=n_bins, range=(lo, hi))
    cdf = np.cumsum(counts)
    target = (n + 1) // 2
    mid = int(np.searchsorted(cdf, target, side="left"))
    mid = min(max(mid, 0), n_bins - 1)
    med = np.float32(0.5 * (edges[mid] + edges[mid + 1]))
    abs_dev = np.abs(row - med)
    dhi = float(np.max(abs_dev))
    if not np.isfinite(dhi) or dhi <= 0:
        return med, np.float32(0)
    counts2, edges2 = np.histogram(abs_dev, bins=n_bins, range=(0.0, dhi))
    cdf2 = np.cumsum(counts2)
    mid2 = int(np.searchsorted(cdf2, target, side="left"))
    mid2 = min(max(mid2, 0), n_bins - 1)
    mad = np.float32(0.5 * (edges2[mid2] + edges2[mid2 + 1]))
    return med, mad


def _pocket_median_mad_auto(row):
    # Exact partition median when DRUGCLIP_CASCADE_ANCHOR_EXACT=1, else histogram.
    if _cascade_anchor_exact():
        return _pocket_median_mad(row)
    return _pocket_median_mad_histogram(row, _cascade_anchor_hist_bins())


class _AnchorReservoir:
    # Fixed random subsample of per-pocket scores collected during tier-0 encode.
    # Avoids a separate full-library memmap scan for median/MAD anchors.
    def __init__(self, n_pock, n_mols, sample_size=262144, seed=1):
        self.n_pock = int(n_pock)
        self.n_mols = int(n_mols)
        self.sample_size = max(0, min(int(sample_size), self.n_mols))
        if self.sample_size <= 0:
            self.sample_idx = np.empty(0, dtype=np.int64)
            self.buf = np.empty((self.n_pock, 0), dtype=np.float32)
        else:
            rng = np.random.RandomState(int(seed))
            self.sample_idx = np.sort(
                rng.choice(self.n_mols, size=self.sample_size, replace=False).astype(
                    np.int64
                )
            )
            self.buf = np.empty((self.n_pock, self.sample_size), dtype=np.float32)
        self._pos = 0

    def add_batch(self, score, offset):
        # score: (n_pock, batch_len) float32; fill buf columns whose global
        # molecule index falls inside [offset, offset + batch_len).
        if self.sample_size == 0 or self._pos >= self.sample_size:
            return
        end = int(offset) + int(score.shape[1])
        while self._pos < self.sample_size and self.sample_idx[self._pos] < end:
            idx = int(self.sample_idx[self._pos])
            if idx >= offset:
                self.buf[:, self._pos] = score[:, idx - offset]
            self._pos += 1

    def finalize(self):
        # Returns (medians, mads) with shape (n_pock, 1) float32.
        medians = np.empty((self.n_pock, 1), dtype=np.float32)
        mads = np.empty((self.n_pock, 1), dtype=np.float32)
        if self.sample_size == 0 or self._pos == 0:
            medians.fill(0)
            mads.fill(0)
            return medians, mads
        n_use = min(self._pos, self.sample_size)
        for p in range(self.n_pock):
            med, mad = _pocket_median_mad_auto(self.buf[p, :n_use])
            medians[p, 0] = med
            mads[p, 0] = mad
        return medians, mads


def _robust_pocket_anchors(scores):
    # Per-pocket median and MAD over molecules. Threaded across pockets: numpy
    # releases the GIL in median/partition, so this scales on multi-core hosts.
    # Always uses exact partition median — histogram is only for gate reservoirs /
    # memmap fallback. Applying histogram here on the dense cascade rescore pool
    # (N > 1M, hundreds of pockets) caused multi-minute memory thrash vs ~20s.
    # Cap workers for large rows so parallel row copies do not swap.
    # Returns (medians, mads) with shape (n_pock, 1) float32.
    # For memmap / out-of-core scores use `_robust_pocket_anchors_memmap` instead.
    scores = np.asarray(scores)
    n_pock = scores.shape[0]
    n_mols = scores.shape[1] if scores.ndim > 1 else 0
    medians = np.empty((n_pock, 1), dtype=np.float32)
    mads = np.empty((n_pock, 1), dtype=np.float32)
    if n_pock == 0:
        return medians, mads

    def _one(p):
        return p, _pocket_median_mad(scores[p])

    if n_pock == 1:
        _, (med, mad) = _one(0)
        medians[0, 0] = med
        mads[0, 0] = mad
        return medians, mads

    cpu = os.cpu_count() or 8
    # Each worker copies ~1-2 full rows (partition + abs_dev). On large dense
    # pools (cascade final rank: 400 x ~1-2M) parallel copies thrash into
    # multi-minute swap under post-encode memory pressure — use sequential.
    row_bytes = max(int(n_mols), 1) * 4
    if n_mols > 500_000 or _cascade_score_bytes(n_pock, n_mols, np.float32) > (
        2 * 1024 * 1024 * 1024
    ):
        n_workers = 1
    else:
        # Keep concurrent row working sets under ~64 MiB (~3 buffers/worker).
        max_by_mem = max(1, (64 * 1024 * 1024) // max(row_bytes * 3, 1))
        n_workers = max(1, min(n_pock, cpu, max_by_mem))

    if n_workers == 1:
        for p in range(n_pock):
            med, mad = _pocket_median_mad(scores[p])
            medians[p, 0] = med
            mads[p, 0] = mad
        return medians, mads

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for p, (med, mad) in pool.map(_one, range(n_pock)):
            medians[p, 0] = med
            mads[p, 0] = mad
    return medians, mads


def _robust_pocket_anchors_memmap(memmap):
    # Sequential per-row median/MAD for on-disk (or huge) score matrices.
    # Never thread across memmap rows: concurrent row reads thrash the page cache.
    # Default uses histogram approximation; DRUGCLIP_CASCADE_ANCHOR_EXACT=1 forces
    # partition-based exact stats.
    n_pock, n_mols = memmap.shape
    medians = np.empty((n_pock, 1), dtype=np.float32)
    mads = np.empty((n_pock, 1), dtype=np.float32)
    for p in range(n_pock):
        row = np.asarray(memmap[p, :], dtype=np.float32)
        med, mad = _pocket_median_mad_auto(row)
        medians[p, 0] = med
        mads[p, 0] = mad
    return medians, mads


def _zscore_chunk_size(n_pock, chunk_size=None):
    # Default working-set for memmap / out-of-core score scans (~64 MiB).
    if chunk_size is not None:
        return max(1, int(chunk_size))
    by_mem = (64 * 1024 * 1024) // (max(int(n_pock), 1) * 4)
    return max(4096, min(_SCORE_MEMMAP_SELECT_CHUNK, by_mem))


def _dense_score_chunk_size(n_pock):
    # Larger molecule-column chunk for dense in-RAM score matrices (~256 MiB).
    # Avoids Python-loop overhead on hundreds of tiny slices without allocating
    # a full (n_pock, N) temporary.
    return max(
        65536,
        min(
            _SCORE_MEMMAP_SELECT_CHUNK,
            (256 * 1024 * 1024) // (max(int(n_pock), 1) * 4),
        ),
    )


def _max_zscore_metric(scores, medians, mads, chunk_size=None):
    # Molecule-chunked robust z-score then max-over-pockets. Avoids allocating a
    # full (n_pock, N) z-scored temporary (critical for hundreds of pockets).
    # Accepts dense arrays or memmap; chunks are cast to float32.
    n_pock, n_mols = scores.shape
    chunk_size = _zscore_chunk_size(n_pock, chunk_size)
    out = np.empty(n_mols, dtype=np.float32)
    medians = np.asarray(medians, dtype=np.float32)
    mads = np.asarray(mads, dtype=np.float32)
    for start in range(0, n_mols, chunk_size):
        end = min(n_mols, start + chunk_size)
        block = np.asarray(scores[:, start:end], dtype=np.float32)
        z = 0.6745 * (block - medians) / (mads + 1e-6)
        out[start:end] = np.max(z, axis=0)
    return out


def _batch_gate_metric(score, medians=None, mads=None):
    # Per-molecule gate metric from one (n_pock, batch) fp32 score block.
    # Matches chunked `_max_zscore_metric` / max-over-pockets when anchors are set.
    score = np.asarray(score, dtype=np.float32)
    if medians is not None:
        medians = np.asarray(medians, dtype=np.float32)
        mads = np.asarray(mads, dtype=np.float32)
        z = 0.6745 * (score - medians) / (mads + 1e-6)
        return np.max(z, axis=0).astype(np.float32, copy=False)
    if score.shape[0] == 1:
        return np.asarray(score[0], dtype=np.float32)
    return np.max(score, axis=0).astype(np.float32, copy=False)


def _max_zscore_metric_from_folds(fold_arrays, medians, mads, chunk_size=None):
    # Chunked fold-mean + robust z-score + max-over-pockets without allocating a
    # full (n_pock, N) mean matrix. Peak temp is one dense chunk (~256 MiB), not
    # an extra multi-GiB combined score matrix on top of the live fold arrays.
    fold_list = list(fold_arrays)
    if not fold_list:
        raise ValueError("fold_arrays must be non-empty")
    n_pock, n_mols = fold_list[0].shape
    n_folds = len(fold_list)
    if chunk_size is None:
        chunk_size = _dense_score_chunk_size(n_pock)
    else:
        chunk_size = max(1, int(chunk_size))
    out = np.empty(n_mols, dtype=np.float32)
    medians = np.asarray(medians, dtype=np.float32)
    mads = np.asarray(mads, dtype=np.float32)
    for start in range(0, n_mols, chunk_size):
        end = min(n_mols, start + chunk_size)
        block = np.asarray(fold_list[0][:, start:end], dtype=np.float32).copy()
        for fa in fold_list[1:]:
            block += np.asarray(fa[:, start:end], dtype=np.float32)
        if n_folds > 1:
            block /= np.float32(n_folds)
        z = 0.6745 * (block - medians) / (mads + 1e-6)
        out[start:end] = np.max(z, axis=0)
    return out


def _gather_score_columns(scores, sorted_idx, chunk_size=None):
    # Gather columns at sorted molecule indices into a dense (n_pock, n_out)
    # float32 array via a sequential column scan. Avoids fancy-index copies that
    # materialize huge temporaries (and thrash) on large (n_pock, N) matrices.
    sorted_idx = np.asarray(sorted_idx, dtype=np.int64).reshape(-1)
    n_pock, n_mols = scores.shape
    n_out = int(sorted_idx.size)
    if n_out == 0:
        return np.empty((n_pock, 0), dtype=np.float32)
    if np.any(sorted_idx[1:] < sorted_idx[:-1]):
        sorted_idx = np.sort(sorted_idx)

    # Small dense gathers: direct indexing is fine and faster. Require both
    # source and output to fit in the 512 MiB budget — fancy-indexing a multi-GiB
    # source (even into a smaller output) thrashs on C-order column gathers.
    is_memmap = isinstance(scores, np.memmap)
    _fast_bytes = 512 * 1024 * 1024
    if (
        not is_memmap
        and _cascade_score_bytes(n_pock, n_mols, np.float32) <= _fast_bytes
        and _cascade_score_bytes(n_pock, n_out, np.float32) <= _fast_bytes
    ):
        return np.asarray(scores[:, sorted_idx], dtype=np.float32)

    chunk_size = _zscore_chunk_size(n_pock, chunk_size)
    out = np.empty((n_pock, n_out), dtype=np.float32)
    out_pos = 0
    idx_pos = 0
    for start in range(0, n_mols, chunk_size):
        if idx_pos >= n_out:
            break
        end = min(n_mols, start + chunk_size)
        chunk_end_idx = idx_pos
        while chunk_end_idx < n_out and sorted_idx[chunk_end_idx] < end:
            chunk_end_idx += 1
        if chunk_end_idx == idx_pos:
            continue
        block = np.asarray(scores[:, start:end], dtype=np.float32)
        local = sorted_idx[idx_pos:chunk_end_idx] - start
        n_take = chunk_end_idx - idx_pos
        out[:, out_pos:out_pos + n_take] = block[:, local]
        out_pos += n_take
        idx_pos = chunk_end_idx
    return out


def _cascade_tier0_select_gather(
    scores, medians, mads, n_pool, do_zscore, chunk_size=None, metric=None
):
    # Memmap-friendly top-k:
    #   - If `metric` is provided (filled during encode): argpartition + one
    #     sequential gather (avoids a second full memmap read).
    #   - Else: chunked metric into a (N,) vector + argpartition + gather.
    # Streaming top-k that kept (n_pock, n_pool) columns in RAM and
    # np.concatenate'd them every chunk was far slower at 400 x ~2M (multi-GB
    # copies / thrash) than a second sequential read.
    n_pock, n_mols = scores.shape
    n_pool = max(1, min(int(n_pool), int(n_mols)))
    # Dense-sized chunks for memmap gather / metric fallback (fewer Python loops
    # than the 64 MiB out-of-core default).
    if chunk_size is None:
        chunk_size = _dense_score_chunk_size(n_pock)
    else:
        chunk_size = max(1, int(chunk_size))

    if metric is not None:
        metric = np.asarray(metric, dtype=np.float32).reshape(-1)
        if int(metric.shape[0]) != int(n_mols):
            raise ValueError(
                f"metric length {metric.shape[0]} does not match n_mols={n_mols}"
            )
    elif do_zscore:
        metric = _max_zscore_metric(
            scores, medians, mads, chunk_size=chunk_size
        )
    elif n_pock == 1:
        metric = np.asarray(scores[0], dtype=np.float32)
    else:
        metric = np.empty(n_mols, dtype=np.float32)
        for start in range(0, n_mols, chunk_size):
            end = min(n_mols, start + chunk_size)
            block = np.asarray(scores[:, start:end], dtype=np.float32)
            metric[start:end] = np.max(block, axis=0)

    sel = np.argpartition(metric, -n_pool)[-n_pool:]
    cur_idx = np.sort(sel)
    del metric
    del sel
    gathered = _gather_score_columns(scores, cur_idx, chunk_size=chunk_size)
    return cur_idx, gathered


def _scratch_st_dev(path):
    # Filesystem device id for scratch-path collision warnings.
    try:
        return os.stat(path).st_dev
    except OSError:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        try:
            return os.stat(parent).st_dev
        except OSError:
            return None


def _log_cascade_scratch(label, path):
    abspath = os.path.abspath(path)
    dev = _scratch_st_dev(abspath)
    logger.info(
        f"cascade scratch {label}: path={abspath} st_dev={dev}"
    )
    return abspath, dev


class _RunningTopK:
    """Streaming top-k over a 1-d metric. Keeps (metric, index) only.

    Used by the quantized-index scan: one heap per target over the
    max-over-pockets z-score, not per-pocket heaps and not score matrices.
    """

    def __init__(self, k):
        self.k = max(1, int(k))
        self.metrics = np.empty(0, dtype=np.float32)
        self.indices = np.empty(0, dtype=np.int64)
        self._buf_m = []
        self._buf_i = []
        self._buf_n = 0
        self._flush_at = max(self.k * 2, 8192)

    def add_batch(self, metrics, indices):
        metrics = np.asarray(metrics, dtype=np.float32).reshape(-1)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if metrics.size == 0:
            return
        if metrics.size != indices.size:
            raise ValueError("metrics and indices must have the same length")
        self._buf_m.append(metrics)
        self._buf_i.append(indices)
        self._buf_n += int(metrics.size)
        if self._buf_n >= self._flush_at:
            self._flush()

    def _flush(self):
        if self._buf_n == 0 and self.metrics.size == 0:
            return
        parts_m = [self.metrics] if self.metrics.size else []
        parts_i = [self.indices] if self.indices.size else []
        parts_m.extend(self._buf_m)
        parts_i.extend(self._buf_i)
        metrics = (
            np.concatenate(parts_m, axis=0) if parts_m else np.empty(0, dtype=np.float32)
        )
        indices = (
            np.concatenate(parts_i, axis=0) if parts_i else np.empty(0, dtype=np.int64)
        )
        self._buf_m.clear()
        self._buf_i.clear()
        self._buf_n = 0
        n = int(metrics.size)
        if n > self.k:
            sel = np.argpartition(metrics, -self.k)[-self.k :]
            metrics = metrics[sel]
            indices = indices[sel]
        self.metrics = metrics
        self.indices = indices

    def finalize(self):
        self._flush()
        if self.indices.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        order = np.argsort(self.metrics, kind="stable")[::-1]
        return self.indices[order], self.metrics[order]

