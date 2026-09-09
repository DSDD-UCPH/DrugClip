#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only diagnostic for the fold-0 cascade gate.

Scores the full molecule library through every fold once, builds the
full-ensemble final ranking (same streamed mean / median-MAD / z-max path as
retrieval), and reports what fraction of the full-run top-1% is captured by the
fold-0 top-`frac` gate for a few candidate fractions. Use this to choose a safe
--cascade-frac before relying on `--retrieval-mode cascade`.

This script never writes molecule embedding caches and does not modify retrieval
output; it only prints (and optionally saves) a small recall report.

For production tightness (~1e-4 of the library, fold-4 gate vs 6-fold top-N
inside top-10N) use ``unimol/validate_fold_gate.py`` instead. This diagnostic
covers the looser cascade_frac=0.2 regime.
"""

import argparse
import logging
import os
import sys
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    logger.info("loading model(s) from {}".format(args.path))
    task = tasks.setup_task(args)
    model = task.build_model(args)

    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()

    logger.info(args)

    model.eval()

    fracs = tuple(float(x) for x in args.fracs.split(",") if x.strip())
    retrieval_bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else None
    save_path = args.report_path if args.report_path else None

    task.cascade_recall_report(
        model,
        args.pocket_path,
        args.mol_path,
        fold_version=args.fold_version,
        fracs=fracs,
        use_cuda=use_cuda,
        save_path=save_path,
        retrieval_bsz=retrieval_bsz,
    )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--fold-version", type=str, default="6_folds", help="fold version")
    parser.add_argument("--fracs", type=str, default="0.05,0.1,0.2", help="comma-separated fold-0 gate fractions to evaluate")
    parser.add_argument("--retrieval-bsz", type=int, default=0, help="DataLoader batch size for molecule scoring; 0 uses the internal default (64)")
    parser.add_argument("--report-path", type=str, default="", help="optional path to write the recall report; empty prints only")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
