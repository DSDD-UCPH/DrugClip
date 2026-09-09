#!/usr/bin/env bash
# Scan 4-bit shard codes then exact all-fold rerank of the shortlist.
# Comma-separate multiple pocket LMDBs; they share one sequential scan.

MANIFEST="./index/manifest.json"
TURBOQUANT="./index/turboquant.npz"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
OUT_DIR="./index/runs/net"
TOPK=2000000
RETRIEVAL_BSZ=256

CUDA_VISIBLE_DEVICES="0" python ./unimol/retrieve_index.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --manifest $MANIFEST \
       --turboquant-path $TURBOQUANT \
       --pocket-path $POCKET_PATH \
       --out-dir $OUT_DIR \
       --topk $TOPK \
       --retrieval-bsz $RETRIEVAL_BSZ
