#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""One-session quantized-index retrieval: scan all shard codes, then exact rerank.

Batch every target pocket-LMDB into a single I/O-bound scan; additional targets
are nearly free. Gate scores and rerank scores are never merged.
"""

import logging
import os
import sys

import torch
from unicore import distributed_utils, options
from unicore import tasks

from unimol.rerank_shortlist import rerank_shortlist
from unimol.scan_index import run_scan

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.retrieve_index")


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

    written, man, _tq, targets = run_scan(args, task, model, use_cuda, use_fp16)
    bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else 256
    fold_version = man.get("fold_version", args.fold_version)
    pocket_paths = [p.strip() for p in args.pocket_path.split(",") if p.strip()]
    for short_path, ppath in zip(written, pocket_paths):
        out = os.path.splitext(short_path)[0] + ".rerank.csv"
        logger.info(f"exact rerank {short_path} -> {out}")
        rerank_shortlist(
            task,
            model,
            short_path,
            ppath,
            man,
            use_cuda,
            use_fp16,
            bsz,
            fold_version,
            use_length_bucket=False,
            write_path=out,
            write_k=args.write_k if args.write_k and args.write_k > 0 else None,
        )


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
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--write-k", type=int, default=0)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
