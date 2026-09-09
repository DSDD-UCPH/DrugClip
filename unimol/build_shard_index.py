#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fused per-shard index build: fold-4 TurboQuant codes + packed LMDB.

Resumable / idempotent: completed codes.npy and packed LMDBs are skipped.
After verifying counts and a checksum, optionally deletes the pickle source.
"""

import hashlib
import logging
import os
import sys

import numpy as np
import torch
from unicore import distributed_utils, options
from unicore import tasks

from unimol.index_encode import encode_fold_embeddings, load_fold_encoder
from unimol.index_manifest import file_checksum, write_shard_sidecar
from unimol.packed_lmdb import PackedLMDBDataset, pack_lmdb_file
from unimol.turboquant import TurboQuant, write_codes_chunked

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.build_shard_index")


def _npy_ok(path, n_mols, width):
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return arr.shape == (int(n_mols), int(width)) and arr.dtype == np.uint8


def _lmdb_n(path, dictionary):
    if not os.path.isfile(path):
        return -1
    ds = PackedLMDBDataset(path, dictionary)
    n = len(ds)
    ds.close()
    return n


def _write_codes(path, embs, tq):
    write_codes_chunked(path, embs, tq, chunk=65536)


def _verify(codes_path, packed_path, n_mols, dictionary, tq):
    arr = np.load(codes_path, mmap_mode="r")
    if arr.shape != (int(n_mols), int(tq.code_width)):
        raise RuntimeError(
            f"codes shape {arr.shape} != ({n_mols}, {tq.code_width})"
        )
    ds = PackedLMDBDataset(packed_path, dictionary)
    try:
        if len(ds) != int(n_mols):
            raise RuntimeError(f"packed LMDB len {len(ds)} != {n_mols}")
        first = ds[0]
        last = ds[n_mols - 1]
        for rec in (first, last):
            if "atoms" not in rec or "coordinates" not in rec or "smi" not in rec:
                raise RuntimeError("packed record missing atoms/coordinates/smi")
    finally:
        ds.close()
    chk = file_checksum(codes_path)
    h = hashlib.sha256()
    h.update(arr[: min(256, n_mols)].tobytes())
    h.update(arr[-1:].tobytes())
    return chk, h.hexdigest()


def main(args):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    tq = TurboQuant.load(args.turboquant_path)
    if tq.dim != 128 or tq.bits != 4:
        raise ValueError(f"unexpected turboquant {tq.metadata()}")

    src_lmdb = os.path.abspath(args.lmdb)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(src_lmdb) or ".")
    os.makedirs(out_dir, exist_ok=True)
    stem = args.stem or os.path.splitext(os.path.basename(src_lmdb))[0]
    codes_path = os.path.join(out_dir, f"{stem}.codes.npy")
    packed_path = os.path.join(out_dir, f"{stem}.packed.lmdb")
    if args.packed_lmdb:
        packed_path = os.path.abspath(args.packed_lmdb)
    if args.codes:
        codes_path = os.path.abspath(args.codes)

    task = tasks.setup_task(args)
    model = task.build_model(args)
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()
    model.eval()

    mol_dataset = task.load_mols_dataset(
        src_lmdb, "atoms", "coordinates", readahead=True
    )
    n_mols = len(mol_dataset)
    logger.info(f"shard {args.shard_id}: {n_mols} mols from {src_lmdb}")

    bsz = args.retrieval_bsz if args.retrieval_bsz and args.retrieval_bsz > 0 else 256
    need_encode = not _npy_ok(codes_path, n_mols, tq.code_width)
    if need_encode:
        mol_model, mol_project = load_fold_encoder(
            task, model, int(args.gate_fold), args.fold_version, use_cuda, use_fp16
        )
        embs = encode_fold_embeddings(
            task,
            mol_model,
            mol_project,
            mol_dataset,
            use_cuda,
            bsz,
            use_length_bucket=bool(args.length_bucket),
            run_label=f"shard {args.shard_id} fold {args.gate_fold}",
        )
        _write_codes(codes_path, embs, tq)
        del embs
        logger.info(f"wrote {codes_path}")
    else:
        logger.info(f"skip encode; {codes_path} already complete")

    task._close_mol_lmdb_envs(mol_dataset)
    del mol_dataset

    packed_ok = _lmdb_n(packed_path, task.dictionary) == n_mols
    if not packed_ok:
        if os.path.abspath(packed_path) == src_lmdb:
            raise ValueError("packed LMDB path must differ from the source pickle LMDB")
        n_written = pack_lmdb_file(src_lmdb, packed_path, task.dictionary)
        if n_written != n_mols:
            raise RuntimeError(f"packed {n_written} != {n_mols}")
        logger.info(f"wrote {packed_path} ({n_written} records)")
    else:
        logger.info(f"skip pack; {packed_path} already complete")

    chk, sample_hash = _verify(
        codes_path, packed_path, n_mols, task.dictionary, tq
    )
    rec, side = write_shard_sidecar(
        codes_path,
        packed_path,
        n_mols,
        int(args.shard_id),
        tq.version,
        extra={"sample_hash": sample_hash, "source_lmdb": src_lmdb},
    )
    rec["checksum"] = chk
    logger.info(f"verified shard {args.shard_id} checksum={chk} sidecar={side}")

    if args.delete_old:
        if os.path.abspath(src_lmdb) == os.path.abspath(packed_path):
            logger.warning("refusing to delete source: same path as packed LMDB")
        else:
            os.remove(src_lmdb)
            logger.info(f"deleted source {src_lmdb}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--lmdb", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--stem", type=str, default="")
    parser.add_argument("--codes", type=str, default="")
    parser.add_argument("--packed-lmdb", type=str, default="")
    parser.add_argument("--turboquant-path", type=str, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--fold-version", type=str, default="6_folds")
    parser.add_argument("--gate-fold", type=int, default=4)
    parser.add_argument("--retrieval-bsz", type=int, default=256)
    parser.add_argument("--length-bucket", type=int, default=1)
    parser.add_argument("--delete-old", action="store_true")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
