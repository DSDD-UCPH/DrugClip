#!/usr/bin/env bash
# Multi-target virtual screening via unimol/retrieval.py.
#
# Expects a folder of pocket LMDBs named <TARGET_NAME>.lmdb. Each file is
# screened independently against the molecule library; results are written to
# <SAVE_DIR>/<TARGET_NAME>.txt.
#
# use_cache=True by default (reuse pre-encoded mol embeddings). After each
# target finishes, per-target pocket embedding and score-memmap caches are
# removed so disk does not accumulate across many targets. Shared mol embedding
# pickles under ./data/encoded_mol_embs/<fold_version>/fold*.pkl are kept.
#
# Usage:
#   bash retrieval_multi_target.sh /path/to/pocket_lmdbs
#   POCKET_DIR=... SAVE_DIR=... bash retrieval_multi_target.sh

set -euo pipefail

MOL_PATH="${MOL_PATH:-mols.lmdb}"
POCKET_DIR="${1:-${POCKET_DIR:-./data/targets}}"
SAVE_DIR="${SAVE_DIR:-./retrieval_results}"
FOLD_VERSION="${FOLD_VERSION:-6_folds}"
use_cache="${use_cache:-True}"

# Retrieval mode:
#   full    - encode SCREEN_FOLDS (default 1,4,5) over the whole library
#   cascade - multi-tier gating then full-fold rescore on survivors
RETRIEVAL_MODE="${RETRIEVAL_MODE:-full}"
SCREEN_FOLDS="${SCREEN_FOLDS:-1,4,5}"
STORE_ALL="${STORE_ALL:-False}"
CASCADE_FRAC="${CASCADE_FRAC:-0.2}"
CASCADE_TIER_FRACS="${CASCADE_TIER_FRACS:-1.0,0.5,0.25}"
CASCADE_GATE_FOLDS="${CASCADE_GATE_FOLDS:-4,1}"

# Persist pocket embeddings / score memmaps during the run so a crashed target
# can resume. Cleared after each successful target (see cleanup below).
WRITE_CACHE="${WRITE_CACHE:-True}"
RETRIEVAL_BSZ="${RETRIEVAL_BSZ:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [ ! -d "$POCKET_DIR" ]; then
  echo "error: pocket directory not found: $POCKET_DIR" >&2
  exit 1
fi

POCKET_LMDBS=()
while IFS= read -r _pocket; do
  [ -n "$_pocket" ] || continue
  POCKET_LMDBS+=("$_pocket")
done <<EOF
$(find "$POCKET_DIR" -maxdepth 1 -type f -name '*.lmdb' | sort)
EOF

if [ "${#POCKET_LMDBS[@]}" -eq 0 ]; then
  echo "error: no *.lmdb files in $POCKET_DIR" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

# Mirror DrugCLIPTask._input_file_cache_tag so we can remove the dirs that
# retrieval wrote for this pocket LMDB.
cache_tag_for() {
  python - "$1" <<'PY'
import hashlib, os, sys
path = sys.argv[1]
abspath = os.path.abspath(path)
try:
    st = os.stat(abspath)
    signature = f"{abspath}:{st.st_size}:{int(st.st_mtime)}"
except OSError:
    signature = abspath
digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:16]
base = os.path.basename(os.path.normpath(path)) or "data"
print(f"{base}_{digest}")
PY
}

cleanup_target_cache() {
  local pocket_path="$1"
  local pocket_tag
  pocket_tag="$(cache_tag_for "$pocket_path")"

  local pocket_cache="./data/encoded_pocket_embs/${FOLD_VERSION}/pocket_cache/${pocket_tag}"
  if [ -d "$pocket_cache" ]; then
    echo "cleaning pocket cache: $pocket_cache"
    rm -rf "$pocket_cache"
  fi

  local score_root="./data/encoded_mol_embs/${FOLD_VERSION}/score_memmap"
  if [ -d "$score_root" ]; then
    # Score dirs are named <mol_tag>__<pocket_tag>
    local d
    for d in "$score_root/"*"__${pocket_tag}"; do
      if [ -e "$d" ] && [ -d "$d" ]; then
        echo "cleaning score memmap: $d"
        rm -rf "$d"
      fi
    done
  fi
}

echo "screening ${#POCKET_LMDBS[@]} target(s) from $POCKET_DIR"
echo "results -> $SAVE_DIR  |  use_cache=$use_cache  |  fold=$FOLD_VERSION"

for POCKET_PATH in "${POCKET_LMDBS[@]}"; do
  TARGET_NAME="$(basename "$POCKET_PATH" .lmdb)"
  save_path="${SAVE_DIR}/${TARGET_NAME}.txt"

  echo "============================================================"
  echo "target: $TARGET_NAME"
  echo "pocket: $POCKET_PATH"
  echo "save:   $save_path"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./dict" --valid-subset test \
       --num-workers "$NUM_WORKERS" --ddp-backend=c10d --batch-size 4 \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --mol-path "$MOL_PATH" \
       --pocket-path "$POCKET_PATH" \
       --fold-version "$FOLD_VERSION" \
       --use-cache "$use_cache" \
       --retrieval-mode "$RETRIEVAL_MODE" \
       --screen-folds "$SCREEN_FOLDS" \
       --store-all "$STORE_ALL" \
       --cascade-frac "$CASCADE_FRAC" \
       --cascade-tier-fracs "$CASCADE_TIER_FRACS" \
       --cascade-gate-folds "$CASCADE_GATE_FOLDS" \
       --write-cache "$WRITE_CACHE" \
       --retrieval-bsz "$RETRIEVAL_BSZ" \
       --prefetch-factor "$PREFETCH_FACTOR" \
       --save-path "$save_path"

  cleanup_target_cache "$POCKET_PATH"
  echo "finished $TARGET_NAME"
done

echo "all ${#POCKET_LMDBS[@]} target(s) complete; results in $SAVE_DIR"
