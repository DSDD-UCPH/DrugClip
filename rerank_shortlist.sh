#!/usr/bin/env bash
# Exact all-fold rerank of a scan shortlist. Does not reuse quantized gate scores.

MANIFEST="./index/manifest.json"
SHORTLIST="./index/runs/net/pocket.shortlist.npz"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
OUT_PATH="./index/runs/net/pocket.rerank.csv"
RETRIEVAL_BSZ=256

CUDA_VISIBLE_DEVICES="0" python ./unimol/rerank_shortlist.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --manifest $MANIFEST \
       --shortlist $SHORTLIST \
       --pocket-path $POCKET_PATH \
       --out-path $OUT_PATH \
       --retrieval-bsz $RETRIEVAL_BSZ
