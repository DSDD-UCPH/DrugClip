#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sweep DataLoader batch size / num_workers for the first-run molecule
encoding path and print a throughput table.

This benchmarks the same compute as the uncached branch of
`retrieval_multi_folds` (the `_run_mol_encoding_pass` molecule pass): each
library batch is pushed through `--num-folds` resident copies of the molecule
encoder. Use it to pick the fastest batch size for a given GPU + molecule
library, since the optimum is rarely "the largest batch that fits".

Example:
    CUDA_VISIBLE_DEVICES=0 python ./unimol/benchmark_bsz.py \
        --user-dir ./unimol ./dict --valid-subset test \
        --task drugclip --loss in_batch_softmax --arch drugclip \
        --max-pocket-atoms 511 --fp16 --fp16-init-scale 4 \
        --fp16-scale-window 256 --seed 1 \
        --mol-path mols.lmdb \
        --bsz-sweep 64,128,256,512,1024 \
        --num-workers-sweep 8 \
        --num-folds 6 --max-mols 20000 --warmup-batches 3
"""

import argparse
import logging
import os
import sys

import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.benchmark")


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    task = tasks.setup_task(args)
    model = task.build_model(args)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    logger.info(args)
    model.eval()

    task.benchmark_mol_encoding(
        model,
        args.mol_path,
        use_cuda,
        batch_sizes=args.bsz_sweep,
        num_workers_list=args.num_workers_sweep,
        num_folds=args.num_folds,
        max_mols=args.max_mols,
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
    )


def cli_main():

    def int_list(v):
        return [int(x) for x in str(v).split(",") if x.strip() != ""]

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol lmdb")
    parser.add_argument(
        "--bsz-sweep",
        type=int_list,
        default=[64, 128, 256, 512, 1024],
        help="comma-separated list of batch sizes to try, e.g. 64,128,256,512,1024",
    )
    parser.add_argument(
        "--num-workers-sweep",
        type=int_list,
        default=None,
        help="comma-separated list of DataLoader worker counts to try "
        "(defaults to --num-workers)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=6,
        help="number of resident fold encoders to mimic (matches fold-version)",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        default=20000,
        help="cap on molecules used per config (0 = whole library)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="batches to skip before timing each config",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="timed repeats per config; the best (fastest) run is reported",
    )
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
