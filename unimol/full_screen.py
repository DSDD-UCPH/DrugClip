# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Full-mode screening helpers (fold subset, native top-k, TurboQuant code layout).

Importable without torch / unicore.
"""

import os

import numpy as np

# Full-mode screening defaults (0-based fold indices). Cascade is unchanged.
DEFAULT_FULL_SCREEN_FOLDS = (1, 4, 5)
FULL_SCREEN_TOPK = 100_000
DEFAULT_TQ_BITS = 2
TQ_SEED = 1
# Names kept so existing 2-bit call sites keep working.
TQ2_BITS = DEFAULT_TQ_BITS
TQ2_SEED = TQ_SEED
TQ_BITS_CHOICES = (0, 1, 2, 3, 4)


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


def _tq_root(fold_version, bits=DEFAULT_TQ_BITS):
    # bits=2 stays `tq2/`, so existing codecs and codes remain valid.
    return os.path.join(
        f"./data/encoded_mol_embs/{fold_version}", f"tq{int(bits)}"
    )


def _tq_codec_path(fold_version, bits=DEFAULT_TQ_BITS):
    # One frozen codec (rotation + codebook) per bit width for every LMDB of
    # this fold_version. Codes and names stay under the per-library tag.
    return os.path.join(_tq_root(fold_version, bits), "turboquant.npz")


def _tq_dir(fold_version, mol_tag, bits=DEFAULT_TQ_BITS):
    return os.path.join(_tq_root(fold_version, bits), str(mol_tag))


def _tq_paths(fold_version, mol_tag, screen_folds, bits=DEFAULT_TQ_BITS):
    cache_dir = _tq_dir(fold_version, mol_tag, bits)
    return {
        "dir": cache_dir,
        "names": os.path.join(cache_dir, "names.npz"),
        "codes": {
            int(f): os.path.join(cache_dir, f"fold_{int(f)}.codes.npy")
            for f in screen_folds
        },
    }


def _tq2_root(fold_version):
    return _tq_root(fold_version, bits=2)


def _tq2_codec_path(fold_version):
    return _tq_codec_path(fold_version, bits=2)


def _tq2_dir(fold_version, mol_tag):
    return _tq_dir(fold_version, mol_tag, bits=2)


def _tq2_paths(fold_version, mol_tag, screen_folds):
    return _tq_paths(fold_version, mol_tag, screen_folds, bits=2)


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


def _tq_codes_ok(path, n_mols, code_width):
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


def _tq2_codes_ok(path, n_mols, code_width):
    return _tq_codes_ok(path, n_mols, code_width)


def _tq_complete(paths, n_mols, code_width):
    if not os.path.isfile(paths["names"]):
        return False
    try:
        names = _load_names_npz(paths["names"])
    except Exception:
        return False
    if len(names) != int(n_mols):
        return False
    return all(
        _tq_codes_ok(path, n_mols, code_width)
        for path in paths["codes"].values()
    )


def _tq2_complete(paths, n_mols, code_width):
    return _tq_complete(paths, n_mols, code_width)
