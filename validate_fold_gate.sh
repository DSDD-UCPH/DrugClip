#!/usr/bin/env bash
# Phase 0: fold-4 gate recall at production tightness (exact fp16, no quant).
# Point MOL_PATH at ONE shard. Drop --cpu to run on GPU.

MOL_PATH="mols.lmdb"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
FOLD_VERSION=6_folds
GATE_FOLD=4
N_FRACS="0.0001,0.0005,0.001,0.01"
GATE_MULT=10
RETRIEVAL_BSZ=256
REPORT_PATH="fold_gate_recall.txt"

CUDA_VISIBLE_DEVICES="0" python ./unimol/validate_fold_gate.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --gate-fold $GATE_FOLD \
       --n-fracs $N_FRACS \
       --gate-mult $GATE_MULT \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --report-path $REPORT_PATH
