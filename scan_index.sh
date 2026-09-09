#!/usr/bin/env bash
# Scan-only: write shortlist npz per target. Rerank separately with
# unimol/rerank_shortlist.py or use retrieve_index.sh for both.

MANIFEST="./index/manifest.json"
TURBOQUANT="./index/turboquant.npz"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
OUT_DIR="./index/runs/net"
TOPK=2000000

CUDA_VISIBLE_DEVICES="0" python ./unimol/scan_index.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --manifest $MANIFEST \
       --turboquant-path $TURBOQUANT \
       --pocket-path $POCKET_PATH \
       --out-dir $OUT_DIR \
       --topk $TOPK
