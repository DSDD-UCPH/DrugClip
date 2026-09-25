#!/bin/bash
###
# use_cache=True: create-or-reuse mol embeddings under
#   ./data/encoded_mol_embs/<fold_version>/fold{i}.npy (float16 mmap) plus names.npy.
#   If that complete npy set (or a legacy complete fold{i}.pkl set) exists, score
#   from it with no LMDB encode. If not, one fused encode scores the current
#   pockets and writes the npy cache.
# use_cache=False: ignore mol caches. One fused score pass only; do not read or
# write fold* mol files. MOL_PATH must be an LMDB screening library.
#
# Full mode scores SCREEN_FOLDS (default 1,4,5) and writes TurboQuant
# codes under ./data/encoded_mol_embs/<fold_version>/tq${TQ_BITS}/<mol_tag>/.
# TQ_BITS=0 stores no codes. TQ_BITS=2 (default) reuses the existing tq2/ tree.
# The codec (rotation + codebook) is shared at
# ./data/encoded_mol_embs/<fold_version>/tq${TQ_BITS}/turboquant.npz.
# WRITE_CACHE persists pocket embeddings only. Score memmaps are always deleted
# after ranking (scratch only).
###


MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="./data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb"
FOLD_VERSION=6_folds

# Disabled as most screens run once per dataset
use_cache=False 
save_path="Scoring_output.txt"

# Retrieval mode:
#   full    - encode --screen-folds (default 1,4,5) over the whole library,
#             rank with native fold-mean scores, and store TurboQuant codes
#             (TQ_BITS, default 2; 0 disables codes). Writes every native score
#             in LMDB order as index,score (${save_path}.all_scores.txt) and the
#             top 100000 hits as index,name,score (${save_path}).
#   cascade - two-phase screening for the use_cache=False / on-the-fly case:
#             1) run CASCADE_TIER_FRACS single-fold gating tiers (one fold each,
#                folds taken from CASCADE_GATE_FOLDS) to progressively narrow the
#                library, keeping CASCADE_FRAC * tier-multiplier each tier;
#             2) re-score the surviving pool through ALL folds and rank with the
#                same procedure as full mode.
RETRIEVAL_MODE=full

# Full-mode fold set (0-based). Ignored when RETRIEVAL_MODE=cascade.
SCREEN_FOLDS=1,4,5
# TurboQuant bits for full mode: 0 = no codes, 1-4 = packed width (default 2).
TQ_BITS=2

# Cascading parameters
CASCADE_FRAC=0.2
# Per-tier fraction multipliers of CASCADE_FRAC; the count sets how many
# single-fold gating tiers run before the all-fold rescore.
CASCADE_TIER_FRACS=1.0,0.5

# Fold index used by each gating tier (padded with remaining folds if shorter).
CASCADE_GATE_FOLDS=4,1

# Persist pocket embeddings to disk (both modes). Mol fold npy is gated by
# use_cache, not WRITE_CACHE. Score memmaps are always removed after results
# are written; they are scratch only. Optional SSD root:
#   export DRUGCLIP_SCORE_MEMMAP_DIR=/path/to/ssd/scratch/score_memmap
# Pocket caches live under:
#   ./data/encoded_pocket_embs/<fold_version>/pocket_cache/<pocket_file>_<hash>/
WRITE_CACHE=True

# DataLoader batch size for molecule encoding/scoring; 0 uses the internal default
# (384 for full mode, 256 for cascade; single-fold gates scale to ~base*n_folds).
# Raise toward 512 if VRAM allows.
RETRIEVAL_BSZ=1024
# Batches prefetched per DataLoader worker (helps keep the GPU fed).
PREFETCH_FACTOR=4

# Cascade gates: Subset-on-source between tiers. Opt-in late compact before
# rescore (DRUGCLIP_CASCADE_COMPACT=1) writes survivors to a sequential temp
# LMDB under DRUGCLIP_CASCADE_TMP (prefer a fast local NVMe).
# Large multi-pocket cascade streams tier-0 scores to a scratch memmap under
# DRUGCLIP_SCORE_MEMMAP_DIR (else CASCADE_TMP, else system temp). Prefer a
# *separate* NVMe for SCORE_MEMMAP_DIR vs CASCADE_TMP so unlinking the ~GiB
# tier0 memmap does not contend with late compact on the same device.
# Score memmap scratch (full + cascade) can share an SSD via DRUGCLIP_SCORE_MEMMAP_DIR.

# Cascade tier-0 rank_select tunables (defaults favor speed; small gate-order
# drift vs the old exact fp32 path is expected):
#   DRUGCLIP_CASCADE_SCORE_DTYPE=float16   # tier-0 memmap dtype (float32 for old)
#   DRUGCLIP_CASCADE_ANCHOR_SAMPLE=262144  # encode-time anchor subsample; 0=full scan
#   DRUGCLIP_CASCADE_ANCHOR_HIST_BINS=8192 # histogram bins for approx median/MAD
#   DRUGCLIP_CASCADE_ANCHOR_EXACT=0        # 1=partition exact median (regression)
# For cold multi-million libraries on HDD, prefer --num-workers 2..4 (seek storms
# with 8+ workers can erase cascade's FLOP advantage). SSD can keep 8.

# export CUDA_VISIBLE_DEVICES="0" 
python ./unimol/retrieval.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers 8 --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --fold-version $FOLD_VERSION \
       --use-cache $use_cache \
       --retrieval-mode $RETRIEVAL_MODE \
       --screen-folds $SCREEN_FOLDS \
       --tq-bits $TQ_BITS \
       --cascade-frac $CASCADE_FRAC \
       --cascade-tier-fracs $CASCADE_TIER_FRACS \
       --cascade-gate-folds $CASCADE_GATE_FOLDS \
       --write-cache $WRITE_CACHE \
       --retrieval-bsz $RETRIEVAL_BSZ \
       --prefetch-factor $PREFETCH_FACTOR \
       --save-path $save_path
