#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sweep --retrieval-bsz for cascade mode and report the optimum per fold count.

Cascade scales the DataLoader batch size per tier via _cascade_tier_bsz: gate
passes run one fold at gate_bsz = base * n_folds, and the rescore pass runs the
remaining folds at a proportionally larger batch. This script sweeps candidate
base batch sizes for each fold count up to --max-folds (default 6), mimicking
the gate and rescore encode passes so you can set --retrieval-bsz before a
cascade run.

Example:
    CUDA_VISIBLE_DEVICES=0 python ./unimol/benchmark_cascade_bsz.py \
        --user-dir ./unimol ./dict --valid-subset test \
        --task drugclip --loss in_batch_softmax --arch drugclip \
        --max-pocket-atoms 511 --fp16 --fp16-init-scale 4 \
        --fp16-scale-window 256 --seed 1 \
        --mol-path mols.lmdb \
        --base-bsz-sweep 64,128,256,512 \
        --max-folds 6 --max-mols 20000 --warmup-batches 3
"""

import argparse
import logging
import os
import sys

import torch
from unicore import distributed_utils, options
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.benchmark_cascade")


def main(args):

    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    task = tasks.setup_task(args)
    model = task.build_model(args)

    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

    logger.info(args)
    model.eval()

    task.benchmark_cascade_bsz(
        model,
        args.mol_path,
        use_cuda,
        base_bsz_sweep=args.base_bsz_sweep,
        max_folds=args.max_folds,
        cascade_tier_fracs=args.cascade_tier_fracs,
        num_workers_list=args.num_workers_sweep,
        max_mols=args.max_mols,
        warmup_batches=args.warmup_batches,
        repeats=args.repeats,
    )


def cli_main():

    def int_list(v):
        return [int(x) for x in str(v).split(",") if x.strip() != ""]

    def float_list(v):
        return [float(x) for x in str(v).split(",") if x.strip() != ""]

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol lmdb")
    parser.add_argument(
        "--base-bsz-sweep",
        type=int_list,
        default=[64, 128, 256, 512],
        help="comma-separated base --retrieval-bsz values to try",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=6,
        help="sweep fold counts from 1 through this value (default 6)",
    )
    parser.add_argument(
        "--cascade-tier-fracs",
        type=float_list,
        default=None,
        help="gate tier multipliers (count sets n_gate_tiers; default 1.0,0.5,0.25)",
    )
    parser.add_argument(
        "--num-workers-sweep",
        type=int_list,
        default=None,
        help="comma-separated DataLoader worker counts (defaults to --num-workers)",
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
