#!/usr/bin/env bash
# One-time frozen TurboQuant + merge shard sidecars into a manifest.

python ./unimol/init_index.py \
    --turboquant-out ./index/turboquant.npz \
    --seed 1 \
    --n-samples 2000000

# After per-shard jobs have written *.shard.json:
# python ./unimol/init_index.py \
#     --turboquant-path ./index/turboquant.npz \
#     --sidecars './index/shards/*.shard.json' \
#     --manifest-out ./index/manifest.json \
#     --anchor-sample 262144 \
#     --anchor-seed 1
