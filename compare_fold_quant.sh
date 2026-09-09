#!/usr/bin/env bash
# Score an input LMDB with one fold (default 4) using the full ranking path,
# cache those embeddings, store TurboQuant codes for that fold, score the
# quantized embeddings the same way, and write both full outputs plus a
# comparison report.

MOL_PATH="mols.lmdb"
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
FOLD_VERSION=6_folds
GATE_FOLD=4
QUANT_BITS=4
OUT_DIR="./fold_quant_compare"
MAX_MOLS=0
TOP_FRACS="0.0001,0.001,0.01"
RETRIEVAL_BSZ=256
TURBOQUANT_PATH=""
REPORT_PATH=""

CUDA_VISIBLE_DEVICES="0" python ./unimol/compare_fold_quant.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --gate-fold $GATE_FOLD \
       --quant-bits $QUANT_BITS \
       --out-dir $OUT_DIR \
       --max-mols $MAX_MOLS \
       --top-fracs $TOP_FRACS \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --turboquant-path "$TURBOQUANT_PATH" \
       --report-path "$REPORT_PATH"
