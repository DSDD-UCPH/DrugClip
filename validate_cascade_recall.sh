
'''
Read-only diagnostic for the fold-0 cascade gate.

Scores the full molecule library through every fold, then reports how much of
the full-ensemble top-2% is captured by the fold-0 top-FRACS gate. Use this to
pick a safe --cascade-frac before running retrieval with --retrieval-mode cascade.

Point MOL_PATH at the lmdb screening library you intend to screen on-the-fly
(the use_cache=False / cascade case). Drop --cpu to run on the GPU.
'''


MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
FOLD_VERSION=6_folds
FRACS="0.05,0.1,0.2"
RETRIEVAL_BSZ=0
REPORT_PATH="cascade_recall_NET.txt"


CUDA_VISIBLE_DEVICES="1" python ./unimol/validate_cascade_recall.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --cpu \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --fracs $FRACS \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --report-path $REPORT_PATH
