# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manifest + turboquant.npz helpers for the quantized shard index."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

MANIFEST_VERSION = 1
DEFAULT_GATE_FOLD = 4


def file_checksum(path, cap_bytes=1_048_576):
    """sha256 of size + mtime + head/tail bytes. Cheap, catches truncated writes."""
    h = hashlib.sha256()
    st = os.stat(path)
    h.update(str(st.st_size).encode("ascii"))
    h.update(b":")
    h.update(str(int(st.st_mtime)).encode("ascii"))
    with open(path, "rb") as f:
        head = f.read(int(cap_bytes))
        h.update(head)
        if st.st_size > 2 * cap_bytes:
            f.seek(max(0, st.st_size - int(cap_bytes)))
            h.update(f.read(int(cap_bytes)))
    return h.hexdigest()


def empty_manifest(turboquant_path, fold_version="6_folds", gate_fold=DEFAULT_GATE_FOLD):
    return {
        "version": MANIFEST_VERSION,
        "fold_version": fold_version,
        "gate_fold": int(gate_fold),
        "turboquant": os.path.abspath(turboquant_path),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anchor_indices": [],
        "shards": [],
    }


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def save_manifest(manifest, path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def shard_sidecar_path(codes_path):
    return os.path.splitext(codes_path)[0] + ".shard.json"


def write_shard_sidecar(
    codes_path,
    lmdb_path,
    n_mols,
    shard_id,
    turboquant_version,
    extra=None,
):
    rec = {
        "id": int(shard_id),
        "lmdb": os.path.abspath(lmdb_path),
        "codes": os.path.abspath(codes_path),
        "n_mols": int(n_mols),
        "turboquant_version": str(turboquant_version),
        "checksum": file_checksum(codes_path) if os.path.exists(codes_path) else "",
    }
    if extra:
        rec.update(extra)
    path = shard_sidecar_path(codes_path)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")
    return rec, path


def merge_sidecars(sidecar_paths, turboquant_path, fold_version="6_folds", gate_fold=DEFAULT_GATE_FOLD):
    shards = []
    for p in sidecar_paths:
        with open(p, "r") as f:
            shards.append(json.load(f))
    shards.sort(key=lambda r: (int(r["id"]), r.get("codes", "")))
    offset = 0
    for rec in shards:
        rec["offset"] = int(offset)
        offset += int(rec["n_mols"])
    man = empty_manifest(turboquant_path, fold_version=fold_version, gate_fold=gate_fold)
    man["shards"] = shards
    man["n_mols_total"] = int(offset)
    return man


def assign_anchor_indices(manifest, sample_size=262144, seed=1):
    n = int(manifest.get("n_mols_total") or sum(int(s["n_mols"]) for s in manifest["shards"]))
    sample_size = max(0, min(int(sample_size), n))
    if sample_size == 0 or n == 0:
        manifest["anchor_indices"] = []
        return manifest
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=sample_size, replace=False)
    idx.sort()
    manifest["anchor_indices"] = idx.astype(np.int64).tolist()
    manifest["anchor_sample_size"] = int(sample_size)
    manifest["anchor_seed"] = int(seed)
    return manifest


def global_to_shard(manifest, global_idx):
    """Map a global molecule index to (shard_record, local_idx)."""
    global_idx = int(global_idx)
    for rec in manifest["shards"]:
        off = int(rec["offset"])
        n = int(rec["n_mols"])
        if off <= global_idx < off + n:
            return rec, global_idx - off
    raise IndexError(f"global index {global_idx} out of range")


def write_shortlist(path, global_idx, metric, manifest):
    """Persist a scan shortlist: global_idx, metric, shard_id, local_idx."""
    global_idx = np.asarray(global_idx, dtype=np.int64).reshape(-1)
    metric = np.asarray(metric, dtype=np.float32).reshape(-1)
    shards = manifest["shards"]
    n = int(global_idx.size)
    shard_id = np.empty(n, dtype=np.int32)
    local_idx = np.empty(n, dtype=np.int32)
    if n > 0:
        order = np.argsort(global_idx, kind="stable")
        g = global_idx[order]
        sid_sorted = np.empty(n, dtype=np.int32)
        loc_sorted = np.empty(n, dtype=np.int32)
        s_i = 0
        for i in range(n):
            gi = int(g[i])
            while s_i + 1 < len(shards) and gi >= int(shards[s_i]["offset"]) + int(
                shards[s_i]["n_mols"]
            ):
                s_i += 1
            rec = shards[s_i]
            sid_sorted[i] = int(rec["id"])
            loc_sorted[i] = gi - int(rec["offset"])
        shard_id[order] = sid_sorted
        local_idx[order] = loc_sorted
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(
        path,
        global_idx=global_idx,
        metric=metric,
        shard_id=shard_id,
        local_idx=local_idx,
    )


def split_global_indices(manifest, global_indices):
    """Group global indices as {shard_id: (local_idx array, global_idx array)}."""
    global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
    if global_indices.size == 0:
        return {}
    order = np.argsort(global_indices, kind="stable")
    g = global_indices[order]
    out = {}
    shard_i = 0
    shards = manifest["shards"]
    pos = 0
    n_g = g.size
    while pos < n_g and shard_i < len(shards):
        rec = shards[shard_i]
        off = int(rec["offset"])
        end = off + int(rec["n_mols"])
        lo = pos
        while pos < n_g and g[pos] < end:
            pos += 1
        if pos > lo:
            loc = g[lo:pos] - off
            out[int(rec["id"])] = (loc, g[lo:pos])
        if pos < n_g and g[pos] >= end:
            shard_i += 1
    return out
