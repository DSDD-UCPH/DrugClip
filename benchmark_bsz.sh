#!/bin/bash
# Sweep DataLoader batch size / num_workers for the first-run molecule encoding
# path and print a throughput table, so you can pick the fastest batch size for
# a given GPU + molecule library.
#
# Point MOL_PATH at the lmdb screening library you intend to encode on-the-fly
# (i.e. the use_cache=False / first-run case). Drop --cpu so it runs on the GPU.

MOL_PATH="mols.lmdb"        # path to the molecule lmdb (screening library)
BSZ_SWEEP="64,128,256,512,1024"
NUM_WORKERS_SWEEP="8"       # try e.g. "4,8,16" to co-tune the DataLoader
NUM_FOLDS=6                 # match your fold-version (6 or 8)
MAX_MOLS=20000             # molecules timed per config; 0 = whole library
WARMUP_BATCHES=3
REPEATS=1

CUDA_VISIBLE_DEVICES="0" python ./unimol/benchmark_bsz.py --user-dir ./unimol "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --bsz-sweep $BSZ_SWEEP \
       --num-workers-sweep $NUM_WORKERS_SWEEP \
       --num-folds $NUM_FOLDS \
       --max-mols $MAX_MOLS \
       --warmup-batches $WARMUP_BATCHES \
       --repeats $REPEATS
