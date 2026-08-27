#!/bin/bash
# Sweep --retrieval-bsz for cascade mode and print the optimum per fold count
# (1..MAX_FOLDS). Mimics single-fold gate passes and the multi-fold rescore pass
# at the tier batch sizes _retrieval_cascade uses, so you can set RETRIEVAL_BSZ
# before running retrieval with --retrieval-mode cascade.
#
# Point MOL_PATH at the lmdb screening library you intend to screen on-the-fly.

MOL_PATH="mols.lmdb"
BASE_BSZ_SWEEP="64,128,256,512"
MAX_FOLDS=6
# Gate tier count follows CASCADE_TIER_FRACS length (default 3 tiers).
CASCADE_TIER_FRACS="1.0,0.5,0.25"
NUM_WORKERS_SWEEP="8"
MAX_MOLS=20000
WARMUP_BATCHES=3
REPEATS=1

CUDA_VISIBLE_DEVICES="0" python ./unimol/benchmark_cascade_bsz.py --user-dir ./unimol "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --base-bsz-sweep $BASE_BSZ_SWEEP \
       --max-folds $MAX_FOLDS \
       --cascade-tier-fracs $CASCADE_TIER_FRACS \
       --num-workers-sweep $NUM_WORKERS_SWEEP \
       --max-mols $MAX_MOLS \
       --warmup-batches $WARMUP_BATCHES \
       --repeats $REPEATS
