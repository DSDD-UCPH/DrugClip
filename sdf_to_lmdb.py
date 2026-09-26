import argparse
import io
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import lmdb
import zstandard as zstd
from rdkit import Chem
from tqdm import tqdm

COMMIT_EVERY = 5_000_000
# Molecules per worker task when --workers > 1.
DEFAULT_CHUNK_MOLS = 1024
ZSTD_LEVEL = 3


def sdf_mol_to_data(mol):
    """Extract atoms/coords/_Name into a pickled blob, or None to skip."""
    try:
        if mol is None:
            return None
        if mol.GetNumConformers() == 0:
            return None
        if not mol.HasProp("_Name"):
            return None

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = mol.GetConformer().GetPositions()
        smi = mol.GetProp("_Name")

        return pickle.dumps(
            {
                "atoms": atoms,
                "coordinates": [coordinates.astype("float16")],
                "smi": smi,
            },
            protocol=-1,
        )
    except Exception:
        return None


def get_sdf_supplier(sdf_path, sanitize=False):
    """Streaming ForwardSDMolSupplier for plain SDF or .zst-compressed SDF.

    Returns (supplier, closer) where closer() releases any open file handles.
    """
    sdf_path = str(sdf_path)
    removeHs = False

    if sdf_path.endswith(".zst"):
        fh = open(sdf_path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = io.BufferedReader(dctx.stream_reader(fh))
        supplier = Chem.ForwardSDMolSupplier(
            reader,
            removeHs=removeHs,
            sanitize=sanitize,
        )

        def closer():
            try:
                reader.close()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass

        return supplier, closer

    fh = open(sdf_path, "rb")
    supplier = Chem.ForwardSDMolSupplier(
        fh,
        removeHs=removeHs,
        sanitize=sanitize,
    )

    def closer():
        try:
            fh.close()
        except Exception:
            pass

    return supplier, closer


def _open_sdf_text_stream(sdf_path):
    """Yield lines from plain or zstd-compressed SDF as text."""
    sdf_path = str(sdf_path)
    if sdf_path.endswith(".zst"):
        fh = open(sdf_path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = io.TextIOWrapper(
            io.BufferedReader(dctx.stream_reader(fh)),
            encoding="utf-8",
            errors="replace",
        )

        def closer():
            try:
                reader.close()
            except Exception:
                pass
            try:
                fh.close()
            except Exception:
                pass

        return reader, closer

    fh = open(sdf_path, "r", encoding="utf-8", errors="replace")
    return fh, fh.close


def iter_sdf_record_chunks(sdf_path, mols_per_chunk=DEFAULT_CHUNK_MOLS):
    """Stream SDF molecule records ($$$$-delimited) as UTF-8 byte chunks."""
    reader, closer = _open_sdf_text_stream(sdf_path)
    buf = []
    n_in_buf = 0
    try:
        for line in reader:
            buf.append(line)
            if line.rstrip("\r\n") == "$$$$":
                n_in_buf += 1
                if n_in_buf >= mols_per_chunk:
                    yield "".join(buf).encode("utf-8")
                    buf = []
                    n_in_buf = 0
        if buf:
            yield "".join(buf).encode("utf-8")
    finally:
        closer()


def _process_sdf_chunk(payload):
    """Worker: parse one SDF text chunk → (list[bytes], skipped)."""
    chunk_bytes, sanitize = payload
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(chunk_bytes),
        removeHs=False,
        sanitize=sanitize,
    )
    out = []
    skipped = 0
    for mol in supplier:
        if mol is None:
            skipped += 1
            continue
        data = sdf_mol_to_data(mol)
        if data is None:
            skipped += 1
            continue
        out.append(data)
    return out, skipped


def _open_lmdb(output_lmdb):
    output_path = Path(output_lmdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(5e11),
    )


def _wants_whole_file_zstd(output_path):
    name = str(output_path).lower()
    return name.endswith(".zst") or name.endswith(".zstd")


def _lmdb_write_path(output_lmdb):
    """Return (write_path, final_zstd_path_or_None)."""
    output_path = Path(output_lmdb)
    if not _wants_whole_file_zstd(output_path):
        return output_path, None
    # Write plain LMDB next to the archive, then compress into -o.
    tmp = output_path.with_name(output_path.name + ".tmp.lmdb")
    return tmp, output_path


def _compress_file_zstd(src_path, dst_path, level=ZSTD_LEVEL):
    """Stream-compress src into dst with zstd, then remove src."""
    compressor = zstd.ZstdCompressor(level=level)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        with compressor.stream_writer(fout) as writer:
            while True:
                chunk = fin.read(1024 * 1024)
                if not chunk:
                    break
                writer.write(chunk)
    os.remove(src_path)


def _process_serial(sdf_path, output_lmdb, sanitize):
    """Single-process streaming path (default)."""
    env = _open_lmdb(output_lmdb)
    supplier, closer = get_sdf_supplier(sdf_path, sanitize=sanitize)

    global_idx = 0
    skipped = 0
    txn = env.begin(write=True)

    try:
        for mol in tqdm(supplier, desc="SDF→LMDB"):
            if mol is None:
                skipped += 1
                continue

            output = sdf_mol_to_data(mol)
            if output is None:
                skipped += 1
                continue

            txn.put(f"{global_idx}".encode("ascii"), output)
            global_idx += 1

            if global_idx % COMMIT_EVERY == 0:
                txn.commit()
                txn = env.begin(write=True)

        txn.commit()
    finally:
        closer()
        env.close()

    print(f"Total processed: {global_idx}")
    print(f"Skipped: {skipped}")


def _process_parallel(
    sdf_path,
    output_lmdb,
    sanitize,
    workers,
    chunk_mols=DEFAULT_CHUNK_MOLS,
):
    """Parse SDF record chunks in a process pool; one writer for LMDB."""
    env = _open_lmdb(output_lmdb)
    global_idx = 0
    skipped = 0
    txn = env.begin(write=True)

    payloads = (
        (chunk, sanitize)
        for chunk in iter_sdf_record_chunks(sdf_path, mols_per_chunk=chunk_mols)
    )

    try:
        with Pool(processes=workers) as pool:
            # imap (ordered) keeps molecule order matching the SDF.
            iterator = pool.imap(
                _process_sdf_chunk,
                payloads,
                chunksize=1,
            )
            for blobs, chunk_skipped in tqdm(iterator, desc="SDF→LMDB"):
                skipped += chunk_skipped
                for blob in blobs:
                    txn.put(f"{global_idx}".encode("ascii"), blob)
                    global_idx += 1
                    if global_idx % COMMIT_EVERY == 0:
                        txn.commit()
                        txn = env.begin(write=True)
        txn.commit()
    finally:
        env.close()

    print(f"Total processed: {global_idx}")
    print(f"Skipped: {skipped}")


def process_sdf_to_lmdb(
    sdf_path,
    output_lmdb,
    sanitize=False,
    workers=1,
    chunk_mols=DEFAULT_CHUNK_MOLS,
):
    """Read molecules from SDF (full file) and write to LMDB.

    workers=1 (default): single-threaded streaming.
    workers>1: shard SDF on $$$$ into text chunks, parse in a process pool,
    write sequentially from the parent (preserves order).

    If output_lmdb ends with .zst or .zstd, the finished LMDB is whole-file
    compressed with zstd level 3 into that path.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    write_path, zstd_path = _lmdb_write_path(output_lmdb)
    if zstd_path is not None and write_path.exists():
        os.remove(write_path)

    if workers == 1:
        _process_serial(sdf_path, write_path, sanitize)
    else:
        _process_parallel(
            sdf_path,
            write_path,
            sanitize,
            workers=workers,
            chunk_mols=chunk_mols,
        )

    if zstd_path is not None:
        print(f"Compressing LMDB → {zstd_path} (zstd level {ZSTD_LEVEL})")
        _compress_file_zstd(write_path, zstd_path, level=ZSTD_LEVEL)
        # Also remove LMDB lock sidecar if present.
        lock = Path(str(write_path) + "-lock")
        if lock.exists():
            try:
                lock.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SDF to LMDB")
    parser.add_argument(
        "--sdf_path",
        "-i",
        required=True,
        help="Path to input SDF file (.sdf or compressed .sdf.zst)",
    )
    parser.add_argument(
        "--output_lmdb",
        "-o",
        required=True,
        help="Output LMDB path; if it ends with .zst/.zstd, whole-file compress",
    )
    san = parser.add_mutually_exclusive_group()
    san.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Disable RDKit sanitization (default)",
    )
    san.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable RDKit sanitization",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Parse worker processes (default: 1 = single-threaded)",
    )
    parser.add_argument(
        "--chunk-mols",
        type=int,
        default=DEFAULT_CHUNK_MOLS,
        help=f"Molecules per worker chunk when --workers > 1 (default: {DEFAULT_CHUNK_MOLS})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sdf_path):
        raise ValueError("Input SDF file does not exist")

    process_sdf_to_lmdb(
        sdf_path=args.sdf_path,
        output_lmdb=args.output_lmdb,
        sanitize=args.sanitize,
        workers=args.workers,
        chunk_mols=args.chunk_mols,
    )
