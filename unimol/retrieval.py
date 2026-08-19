#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
import pickle
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


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)
        # TF32 speeds the fp32 pocket/score matmuls on Ampere+ with negligible
        # impact on retrieval ranking.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    #state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    #model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()
        logger.info(
            "retrieval device: cuda:{} (fp16={})".format(args.device_id, use_fp16)
        )
    else:
        logger.info("retrieval device: cpu (fp16={})".format(use_fp16))

    # Print args
    logger.info(args)


    model.eval()
    
    #names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)
    print(111, args.use_cache)

    retrieval_bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else None

    task.retrieval_multi_folds(model, args.pocket_path, args.save_path, args.mol_path, fold_version=args.fold_version, use_cache=args.use_cache, use_cuda=use_cuda, retrieval_mode=args.retrieval_mode, cascade_frac=args.cascade_frac, cascade_tier_fracs=args.cascade_tier_fracs, cascade_gate_folds=args.cascade_gate_folds, write_cache=args.write_cache, retrieval_bsz=retrieval_bsz)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--mol-path", type=str, default="", help="path for mol data")
    parser.add_argument("--pocket-path", type=str, default="", help="path for pocket data")
    parser.add_argument("--fold-version", type=str, default="6_folds", help="fold version")
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    def float_list(v):
        return [float(x) for x in str(v).split(",") if x.strip() != ""]

    def int_list(v):
        return [int(x) for x in str(v).split(",") if x.strip() != ""]

    parser.add_argument("--use-cache", type=str2bool, default=False, help="whether use pre-encoded embeddings")
    parser.add_argument("--retrieval-mode", type=str, default="full", choices=["full", "cascade"], help="full: encode every fold over the whole library; cascade: configurable multi-tier gating that scores one fold per tier to progressively narrow the pool, then re-scores the surviving pool through all folds and ranks with the same procedure as full mode")
    parser.add_argument("--cascade-frac", type=float, default=0.2, help="tier-1 fraction of the library kept after the first gate in cascade mode; each tier's kept fraction is this value times the matching --cascade-tier-fracs multiplier")
    parser.add_argument("--cascade-tier-fracs", type=float_list, default=[1.0, 0.5, 0.25], help="comma-separated multipliers of --cascade-frac, one per gating tier (e.g. 1.0,0.5,0.25); the number of entries sets how many single-fold gating tiers run before the full-fold rescore")
    parser.add_argument("--cascade-gate-folds", type=int_list, default=[4, 1], help="comma-separated fold index used by each gating tier (e.g. 4,1); shorter than --cascade-tier-fracs is padded with the remaining unused folds in ascending order")
    parser.add_argument("--write-cache", type=str2bool, default=True, help="whether to persist newly encoded score memmaps (full mode) and pocket embeddings (both modes) to disk")
    parser.add_argument("--retrieval-bsz", type=int, default=0, help="DataLoader batch size for molecule encoding/scoring; 0 uses the internal default (384 for full mode, 64 for cascade)")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch_factor when num_workers > 0 (default 4)")
    parser.add_argument("--save-path", type=str, default="", help="path for saved result")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
