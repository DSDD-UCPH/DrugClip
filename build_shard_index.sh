#!/usr/bin/env bash
# Fused per-shard index build: fold-4 4-bit codes + packed LMDB.
# Idempotent. Add --delete-old only after a verified run.

LMDB="mols.lmdb"
OUT_DIR="./index/shards/0"
SHARD_ID=0
TURBOQUANT="./index/turboquant.npz"
FOLD_VERSION=6_folds
GATE_FOLD=4
RETRIEVAL_BSZ=256

CUDA_VISIBLE_DEVICES="0" python ./unimol/build_shard_index.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --lmdb $LMDB \
       --out-dir $OUT_DIR \
       --shard-id $SHARD_ID \
       --turboquant-path $TURBOQUANT \
       --fold-version $FOLD_VERSION \
       --gate-fold $GATE_FOLD \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --length-bucket 1
