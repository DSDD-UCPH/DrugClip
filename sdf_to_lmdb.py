import argparse
import io
import os
import pickle
from pathlib import Path

import lmdb
import zstandard as zstd
from rdkit import Chem
from tqdm import tqdm

COMMIT_EVERY = 50_000


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


def process_sdf_to_lmdb(
    sdf_path,
    output_lmdb,
    sanitize=False,
    compress=True,
):
    """Read molecules from SDF (full file) and write to LMDB (single thread)."""
    output_path = Path(output_lmdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        writemap=True,
        map_async=True,
        max_readers=1,
        map_size=int(5e11),
    )

    compressor = zstd.ZstdCompressor(level=3) if compress else None
    supplier, closer = get_sdf_supplier(sdf_path, sanitize=sanitize)

    global_idx = 0
    skipped = 0
    txn = env.begin(write=True)

    try:
        for mol in tqdm(supplier):
            if mol is None:
                skipped += 1
                continue

            output = sdf_mol_to_data(mol)
            if output is None:
                skipped += 1
                continue

            if compressor is not None:
                output = compressor.compress(output)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SDF to LMDB")
    parser.add_argument(
        "--sdf_path",
        "-i",
        required=True,
        help="Path to input SDF file (.sdf or .sdf.zst)",
    )
    parser.add_argument(
        "--output_lmdb",
        "-o",
        required=True,
        help="Output LMDB file path",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable RDKit sanitization (slower; off by default)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Store raw pickle values instead of zstd-compressed blobs",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sdf_path):
        raise ValueError("Input SDF file does not exist")

    process_sdf_to_lmdb(
        sdf_path=args.sdf_path,
        output_lmdb=args.output_lmdb,
        sanitize=args.sanitize,
        compress=not args.no_compress,
    )
