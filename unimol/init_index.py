#!/usr/bin/env python3
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Create turboquant.npz and merge per-shard sidecars into a manifest."""

from __future__ import annotations

import argparse
import glob
import os
import sys

from unimol.index_manifest import (
    assign_anchor_indices,
    merge_sidecars,
    save_manifest,
)
from unimol.turboquant import TurboQuant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--turboquant-out",
        type=str,
        default="",
        help="write a new frozen turboquant.npz here",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=2_000_000)
    p.add_argument(
        "--sidecars",
        type=str,
        default="",
        help="glob of *.shard.json files to merge",
    )
    p.add_argument("--manifest-out", type=str, default="")
    p.add_argument("--turboquant-path", type=str, default="")
    p.add_argument("--fold-version", type=str, default="6_folds")
    p.add_argument("--gate-fold", type=int, default=4)
    p.add_argument("--anchor-sample", type=int, default=262144)
    p.add_argument("--anchor-seed", type=int, default=1)
    args = p.parse_args()

    tq_path = args.turboquant_path
    if args.turboquant_out:
        tq = TurboQuant.create(
            dim=128, bits=4, seed=args.seed, n_samples=args.n_samples
        )
        parent = os.path.dirname(os.path.abspath(args.turboquant_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tq.save(args.turboquant_out)
        tq_path = args.turboquant_out
        print(f"wrote {args.turboquant_out} version={tq.version}")

    if args.sidecars:
        if not tq_path:
            sys.exit("--turboquant-path (or --turboquant-out) required with --sidecars")
        if not args.manifest_out:
            sys.exit("--manifest-out required with --sidecars")
        paths = sorted(glob.glob(args.sidecars))
        if not paths:
            sys.exit(f"no sidecars matched {args.sidecars}")
        man = merge_sidecars(
            paths,
            tq_path,
            fold_version=args.fold_version,
            gate_fold=args.gate_fold,
        )
        versions = {s.get("turboquant_version") for s in man["shards"]}
        if len(versions) != 1:
            sys.exit(f"mixed turboquant versions in sidecars: {versions}")
        assign_anchor_indices(man, sample_size=args.anchor_sample, seed=args.anchor_seed)
        save_manifest(man, args.manifest_out)
        print(
            f"wrote {args.manifest_out}: {len(man['shards'])} shards, "
            f"{man['n_mols_total']} mols, "
            f"{len(man['anchor_indices'])} anchors"
        )


if __name__ == "__main__":
    main()
