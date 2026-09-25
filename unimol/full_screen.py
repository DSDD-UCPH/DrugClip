# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Full-mode screening helpers (fold subset, native top-k, 2-bit code layout).

Importable without torch / unicore.
"""

import os

import numpy as np

# Full-mode screening defaults (0-based fold indices). Cascade is unchanged.
DEFAULT_FULL_SCREEN_FOLDS = (1, 4, 5)
FULL_SCREEN_TOPK = 100_000
TQ2_BITS = 2
TQ2_SEED = 1


def _resolve_screen_folds(screen_folds, n_folds, default=DEFAULT_FULL_SCREEN_FOLDS):
    # Unique fold ids in caller order. Empty / None uses the full-mode default.
    if screen_folds is None:
        folds = list(default)
    else:
        folds = [int(f) for f in screen_folds]
    if not folds:
        folds = list(default)
    n_folds = int(n_folds)
    out = []
    seen = set()
    for fold in folds:
        if fold < 0 or fold >= n_folds:
            raise ValueError(
                f"screen fold {fold} out of range for {n_folds} folds"
            )
        if fold not in seen:
            out.append(fold)
            seen.add(fold)
    if not out:
        raise ValueError("screen_folds must be non-empty")
    return out


def _screen_folds_tag(screen_folds):
    return "f" + "-".join(str(int(f)) for f in sorted(int(x) for x in screen_folds))


def _full_screen_k(n_mols, topk=FULL_SCREEN_TOPK):
    n_mols = int(n_mols)
    if n_mols <= 0:
        return 0
    return max(1, min(n_mols, int(topk)))


def _tq2_root(fold_version):
    return os.path.join(f"./data/encoded_mol_embs/{fold_version}", "tq2")


def _tq2_codec_path(fold_version):
    # One frozen 2-bit codec (rotation + codebook) for every LMDB of this
    # fold_version. Codes and names stay under the per-library tag directory.
    return os.path.join(_tq2_root(fold_version), "turboquant.npz")


def _tq2_dir(fold_version, mol_tag):
    return os.path.join(_tq2_root(fold_version), str(mol_tag))


def _tq2_paths(fold_version, mol_tag, screen_folds):
    cache_dir = _tq2_dir(fold_version, mol_tag)
    return {
        "dir": cache_dir,
        "names": os.path.join(cache_dir, "names.npz"),
        "codes": {
            int(f): os.path.join(cache_dir, f"fold_{int(f)}.codes.npy")
            for f in screen_folds
        },
    }


def _write_names_npz(path, names):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    arr = np.asarray(list(names), dtype=str)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, names=arr)
    os.replace(tmp, path)
    return path


def _load_names_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return [str(x) for x in data["names"]]


def _tq2_codes_ok(path, n_mols, code_width):
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return (
        arr.dtype == np.uint8
        and tuple(arr.shape) == (int(n_mols), int(code_width))
    )


def _tq2_complete(paths, n_mols, code_width):
    if not os.path.isfile(paths["names"]):
        return False
    try:
        names = _load_names_npz(paths["names"])
    except Exception:
        return False
    if len(names) != int(n_mols):
        return False
    return all(
        _tq2_codes_ok(path, n_mols, code_width)
        for path in paths["codes"].values()
    )
