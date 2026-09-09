#!/usr/bin/env bash
# Phase 0: 4-bit MSE-only TurboQuant vs exact fold-4 fp16 (Jaccard / MAE / Pearson / MAD).

MOL_PATH="mols.lmdb"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
FOLD_VERSION=6_folds
GATE_FOLD=4
MAX_MOLS=50000
TOP_FRAC=0.01
RETRIEVAL_BSZ=256
TURBOQUANT_PATH=""
SAVE_TURBOQUANT="./index/turboquant.npz"
REPORT_PATH="quant_delta.txt"

CUDA_VISIBLE_DEVICES="0" python ./unimol/validate_quant_delta.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --gate-fold $GATE_FOLD \
       --max-mols $MAX_MOLS \
       --top-frac $TOP_FRAC \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --turboquant-path "$TURBOQUANT_PATH" \
       --save-turboquant $SAVE_TURBOQUANT \
       --report-path $REPORT_PATH
