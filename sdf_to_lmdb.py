"""SDF → LMDB plus a line-aligned SMILES file.

RDKit sanitization is never used. Structures are parsed as written (the
library is already prepared). Canonical isomeric SMILES are still generated
for as many records as possible: explicit hydrogens are removed only on a
copy used for SMILES, valences are accepted with a non-strict property cache,
and aromaticity is perceived without the sanitizer's reject path.

Line ``i`` of the SMILES file (0-based, no header) is LMDB key ``i``.
A source record is written to both outputs or to neither.
The LMDB ``smi`` field stays the SDF title, same as ``sdf_to_lmdb.py``.
The SMILES file is ``SMILES<TAB>title``.

Throughput layout (multi-million-molecule SDFs):
- The parent only finds record-aligned byte ranges. For plain SDFs workers
  read their own range from disk, so no SDF text crosses process pipes.
- At most ``workers * INFLIGHT_PER_WORKER`` chunks are outstanding, so memory
  stays flat regardless of input size (including ``.sdf.zst``).
- Workers parse each record's mol block directly (SD data items are not
  needed) and read atom symbols from the V2000 atom block text, falling back
  to RDKit when the text is not a plain element symbol.
"""

import argparse
import gc
import os
import pickle
from collections import deque
from multiprocessing import get_context
from pathlib import Path

import lmdb
import zstandard as zstd
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

from sdf_to_lmdb import (
    ZSTD_LEVEL,
    _compress_file_zstd,
    _lmdb_write_path,
)

# Parsing warnings dominate runtime on multi-million SDFs and do not
# change which records are kept.
RDLogger.DisableLog("rdApp.*")

# Commits are cheap with sync off. Frequent commits keep LMDB's dirty-page
# list small over multi-million-record runs.
COMMIT_EVERY = 250_000
DEFAULT_CHUNK_MB = 4.0
INFLIGHT_PER_WORKER = 4
_SCAN_WINDOW = 1 << 16
_ZST_READ = 1 << 22

_MOL_FROM_MOL_BLOCK = Chem.MolFromMolBlock
_REMOVE_HS = Chem.RemoveHs
_MOL_TO_SMILES = Chem.MolToSmiles
_FAST_FIND_RINGS = Chem.FastFindRings
_SET_AROMATICITY = Chem.SetAromaticity
_AROMATIC = Chem.BondType.AROMATIC

_PT = Chem.GetPeriodicTable()
# Symbols RDKit returns unchanged from GetSymbol. D, T, R#, A, Q, etc.
# are absent and trigger the RDKit fallback.
_ELEMENTS = frozenset(_PT.GetElementSymbol(z) for z in range(1, 119))

_SKIP_KEYS = ("parse", "record", "smiles")


def _discard_file(path):
    path = Path(path)
    if path.exists():
        path.unlink()
    lock = Path(str(path) + "-lock")
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass


def _open_lmdb(output_lmdb):
    output_path = Path(output_lmdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # sync/metasync off: durability comes from the final env.sync().
    return lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(5e11),
        sync=False,
        metasync=False,
    )


# ---------------------------------------------------------------------------
# Record-aligned chunking (parent process)
# ---------------------------------------------------------------------------


def _record_end_after(buf, start, eof):
    """Offset just past the first '$$$$' line at/after ``start``, else -1."""
    i = buf.find(b"\n$$$$", start)
    if i == -1:
        return -1
    nl = buf.find(b"\n", i + 5)
    if nl != -1:
        return nl + 1
    return len(buf) if eof else -1


def _last_record_end(buf, eof):
    """Offset just past the last complete '$$$$' line in ``buf``, else -1."""
    i = buf.rfind(b"\n$$$$")
    while i != -1:
        nl = buf.find(b"\n", i + 5)
        if nl != -1:
            return nl + 1
        if eof:
            return len(buf)
        i = buf.rfind(b"\n$$$$", 0, i)
    return -1


def iter_plain_spans(sdf_path, chunk_bytes):
    """Yield ``(start, end)`` byte ranges of whole records in a plain SDF.

    Only small windows around each boundary are read here.
    """
    size = os.path.getsize(sdf_path)
    pos = 0
    with open(sdf_path, "rb") as fh:
        while pos < size:
            target = pos + chunk_bytes
            if target >= size:
                yield pos, size
                return
            # Step back one byte so a boundary starting exactly at target
            # is still seen.
            scan = target - 1
            fh.seek(scan)
            buf = b""
            end = -1
            while True:
                block = fh.read(_SCAN_WINDOW)
                eof = not block
                buf += block
                end = _record_end_after(buf, 0, eof)
                if end != -1 or eof:
                    break
                # Keep a tail that may hold a partial "\n$$$$".
                if len(buf) > _SCAN_WINDOW:
                    drop = len(buf) - 5
                    scan += drop
                    buf = buf[drop:]
            if end == -1:
                yield pos, size
                return
            end += scan
            yield pos, end
            pos = end


def iter_zst_chunks(sdf_path, chunk_bytes):
    """Yield record-aligned decompressed byte chunks from a .zst SDF."""
    with open(sdf_path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        buf = bytearray()
        while True:
            block = reader.read(_ZST_READ)
            eof = not block
            buf += block
            while len(buf) >= chunk_bytes or (eof and buf):
                if eof and len(buf) < chunk_bytes:
                    yield bytes(buf)
                    buf.clear()
                    break
                cut = _record_end_after(buf, chunk_bytes - 1, eof)
                if cut == -1:
                    if eof:
                        yield bytes(buf)
                        buf.clear()
                    break
                yield bytes(buf[:cut])
                del buf[:cut]
            if eof:
                return


# ---------------------------------------------------------------------------
# SMILES without sanitization
# ---------------------------------------------------------------------------


def _title_field(name):
    """One TSV field. Tabs and newlines would split the SMILES record."""
    if "\t" in name or "\n" in name or "\r" in name:
        return name.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return name


def _lowercase_aromatic(mol):
    """Write aromatic bonds as lowercase atoms when kekulization is impossible.

    RDKit otherwise emits ':' bonds, which ``MolFromSmiles`` will not read.
    Operates on a copy. Returns an empty string on failure.
    """
    copied = Chem.Mol(mol)
    touched = False
    for bond in copied.GetBonds():
        if bond.GetBondType() == _AROMATIC or bond.GetIsAromatic():
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
            touched = True
    if not touched:
        return ""
    try:
        smiles = _MOL_TO_SMILES(copied, isomericSmiles=True, canonical=True)
    except Exception:
        return ""
    if smiles and ":" not in smiles:
        return smiles
    return ""


def _smiles_from_prepared(mol):
    """Canonical isomeric SMILES, then kekulé and aromatic fallbacks."""
    try:
        primary = _MOL_TO_SMILES(mol, isomericSmiles=True, canonical=True) or ""
    except Exception:
        primary = ""
    if primary and ":" not in primary:
        return primary
    # ':' means aromatic bonds survived without atom aromaticity. Prefer a
    # kekulé SMILES, then a lowercase aromatic SMILES, both of which readers
    # accept more often than ':' bonds.
    try:
        kek = _MOL_TO_SMILES(
            mol, isomericSmiles=True, canonical=True, kekuleSmiles=True
        ) or ""
    except Exception:
        kek = ""
    if kek and ":" not in kek:
        return kek
    lowered = _lowercase_aromatic(mol)
    if lowered:
        return lowered
    if primary:
        return primary
    if kek:
        return kek
    try:
        return (
            _MOL_TO_SMILES(
                mol, isomericSmiles=True, canonical=False, kekuleSmiles=True
            )
            or ""
        )
    except Exception:
        return ""


def _prepare_for_smiles(mol):
    """Make an unsanitized mol writable. Returns False if valence setup fails."""
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        return False
    try:
        _FAST_FIND_RINGS(mol)
    except Exception:
        pass
    try:
        _SET_AROMATICITY(mol)
    except Exception:
        pass
    return True


def mol_to_max_smiles(mol, has_h=None):
    """Canonical isomeric SMILES without SanitizeMol.

    Explicit hydrogens are dropped on a copy so the stored conformer is
    unchanged. Molecules the sanitizer would reject still produce a SMILES
    when the connection table can be written at all. Pass ``has_h`` when
    already known to skip an atom scan.
    """
    if has_h is None:
        has_h = any(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 1
            for i in range(mol.GetNumAtoms())
        )
    candidates = []
    if has_h:
        try:
            stripped = _REMOVE_HS(mol, sanitize=False)
        except Exception:
            stripped = None
        if stripped is not None and stripped.GetNumAtoms() > 0:
            candidates.append(stripped)
    candidates.append(mol)

    for candidate in candidates:
        if not _prepare_for_smiles(candidate):
            continue
        smiles = _smiles_from_prepared(candidate)
        if smiles:
            return smiles
    return ""


# ---------------------------------------------------------------------------
# Per-record conversion (workers)
# ---------------------------------------------------------------------------


def _atom_symbols(block, mol, n_atoms):
    """Atom symbols in file order, identical to ``atom.GetSymbol()``."""
    lines = block.split("\n", 4 + n_atoms)
    if len(lines) > 4 + n_atoms:
        counts = lines[3]
        if "V3000" not in counts:
            try:
                declared = int(counts[0:3])
            except ValueError:
                declared = -1
            if declared == n_atoms:
                symbols = [line[31:34].strip() for line in lines[4 : 4 + n_atoms]]
                if _ELEMENTS.issuperset(symbols):
                    return symbols
    get = mol.GetAtomWithIdx
    return [get(i).GetSymbol() for i in range(n_atoms)]


def _convert_record(text):
    """Return ``(blob, smiles_line)`` or ``(None, skip_reason)``."""
    end = text.find("\nM  END")
    block = text[: end + 7] if end >= 0 else text
    try:
        mol = _MOL_FROM_MOL_BLOCK(
            block, sanitize=False, removeHs=False, strictParsing=False
        )
    except Exception:
        mol = None
    if mol is None:
        return None, "parse"

    try:
        if mol.GetNumConformers() == 0 or not mol.HasProp("_Name"):
            return None, "record"
        n_atoms = mol.GetNumAtoms()
        atoms = _atom_symbols(block, mol, n_atoms)
        name = mol.GetProp("_Name")
        blob = pickle.dumps(
            {
                "atoms": atoms,
                "coordinates": [mol.GetConformer().GetPositions().astype("float16")],
                "smi": name,
            },
            protocol=-1,
        )
    except Exception:
        return None, "record"

    try:
        smiles = mol_to_max_smiles(mol, "H" in atoms)
    except Exception:
        smiles = ""
    if not smiles:
        return None, "smiles"
    return blob, f"{smiles}\t{_title_field(name)}\n"


def split_records(text):
    """Split SDF text into records. RDKit ends a record at any line starting with '$$$$'."""
    records = []
    pos = 0
    if text.startswith("$$$$"):
        nl = text.find("\n")
        if nl == -1:
            return records
        pos = nl + 1
    find = text.find
    while True:
        i = find("\n$$$$", pos)
        if i == -1:
            records.append(text[pos:])
            return records
        records.append(text[pos : i + 1])
        nl = find("\n", i + 5)
        if nl == -1:
            return records
        pos = nl + 1


def process_chunk_bytes(chunk):
    """One record-aligned chunk → (blobs, joined SMILES text, skipped, n_records)."""
    text = chunk.decode("utf-8", errors="replace")
    blobs = []
    lines = []
    skipped = dict.fromkeys(_SKIP_KEYS, 0)
    n_records = 0
    for record in split_records(text):
        if not record or record.isspace():
            continue
        n_records += 1
        blob, second = _convert_record(record)
        if blob is None:
            skipped[second] += 1
            continue
        blobs.append(blob)
        lines.append(second)
    return blobs, "".join(lines), skipped, n_records


_WORKER_FD = None


def _worker_init(sdf_path):
    global _WORKER_FD
    # RDKit mols are acyclic from Python's view; cyclic GC only costs time.
    gc.disable()
    RDLogger.DisableLog("rdApp.*")
    if sdf_path is not None:
        _WORKER_FD = os.open(sdf_path, os.O_RDONLY)


def _process_span(span):
    start, end = span
    return process_chunk_bytes(os.pread(_WORKER_FD, end - start, start))


def _read_span(fd, span):
    start, end = span
    return os.pread(fd, end - start, start)


# ---------------------------------------------------------------------------
# Writer (parent process)
# ---------------------------------------------------------------------------


class _Writer:
    """Append chunk results to LMDB and the SMILES file with one shared index."""

    def __init__(self, output_lmdb, output_smi):
        self.env = _open_lmdb(output_lmdb)
        self.txn = self.env.begin(write=True)
        self.smi = open(
            output_smi, "w", encoding="utf-8", newline="\n", buffering=1 << 22
        )
        self.written = 0
        self.since_commit = 0
        self.skipped = dict.fromkeys(_SKIP_KEYS, 0)

    def add(self, result):
        blobs, smiles_text, skipped, _ = result
        for key, count in skipped.items():
            self.skipped[key] = self.skipped.get(key, 0) + count
        if not blobs:
            return
        start = self.written
        keys = [str(i).encode("ascii") for i in range(start, start + len(blobs))]
        self.txn.cursor().putmulti(zip(keys, blobs))
        self.smi.write(smiles_text)
        self.written += len(blobs)
        self.since_commit += len(blobs)
        if self.since_commit >= COMMIT_EVERY:
            self.txn.commit()
            self.smi.flush()
            self.txn = self.env.begin(write=True)
            self.since_commit = 0

    def close(self, ok):
        try:
            if ok:
                self.txn.commit()
                self.env.sync(True)
            else:
                self.txn.abort()
        finally:
            self.env.close()
            self.smi.close()


def _print_summary(written, skipped):
    skipped_n = sum(skipped.values())
    print(f"Total processed: {written}")
    print(f"SMILES lines: {written}")
    print(f"Skipped: {skipped_n}")
    if skipped_n:
        print(
            "Skipped breakdown: "
            + ", ".join(f"{key}={count}" for key, count in sorted(skipped.items()))
        )


def _run(tasks, handle, writer, workers, sdf_path_for_workers):
    """Feed tasks with bounded in-flight work; write results in order."""
    bar = tqdm(desc="SDF→LMDB+SMI", unit="mol", smoothing=0.05)
    try:
        if workers == 1:
            _worker_init(None)
            for task in tasks:
                result = handle(task)
                writer.add(result)
                bar.update(result[3])
            return

        ctx = get_context("fork")
        max_inflight = workers * INFLIGHT_PER_WORKER
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(sdf_path_for_workers,),
        ) as pool:
            pending = deque()
            for task in tasks:
                pending.append(pool.apply_async(handle, (task,)))
                if len(pending) >= max_inflight:
                    result = pending.popleft().get()
                    writer.add(result)
                    bar.update(result[3])
            while pending:
                result = pending.popleft().get()
                writer.add(result)
                bar.update(result[3])
    finally:
        bar.close()


def process_sdf_to_lmdb_smi(
    sdf_path,
    output_lmdb,
    output_smi,
    workers=None,
    chunk_mb=DEFAULT_CHUNK_MB,
):
    """Write an LMDB and a SMILES file with the same 0-based index.

    workers defaults to the CPU count; workers=1 runs in-process.
    chunk_mb is the approximate SDF text per worker task.
    If output_lmdb ends with .zst or .zstd, the finished LMDB is whole-file
    compressed with zstd level 3.
    """
    if workers is None:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if chunk_mb <= 0:
        raise ValueError("chunk_mb must be > 0")
    chunk_bytes = max(1 << 16, int(chunk_mb * (1 << 20)))

    sdf_path = str(sdf_path)
    smi_path = Path(output_smi)
    if smi_path.resolve() == Path(output_lmdb).resolve():
        raise ValueError("SMILES path and LMDB path must differ")
    smi_path.parent.mkdir(parents=True, exist_ok=True)

    write_path, zstd_path = _lmdb_write_path(output_lmdb)
    _discard_file(write_path)

    compressed = sdf_path.endswith(".zst")
    if compressed:
        tasks = iter_zst_chunks(sdf_path, chunk_bytes)
        handle = process_chunk_bytes
        worker_path = None
    elif workers == 1:
        fd = os.open(sdf_path, os.O_RDONLY)
        tasks = (_read_span(fd, span) for span in iter_plain_spans(sdf_path, chunk_bytes))
        handle = process_chunk_bytes
        worker_path = None
    else:
        fd = None
        tasks = iter_plain_spans(sdf_path, chunk_bytes)
        handle = _process_span
        worker_path = sdf_path

    writer = _Writer(write_path, smi_path)
    ok = False
    try:
        _run(tasks, handle, writer, workers, worker_path)
        ok = True
    finally:
        writer.close(ok)
        if not compressed and workers == 1:
            os.close(fd)
    _print_summary(writer.written, writer.skipped)

    if zstd_path is not None:
        print(f"Compressing LMDB → {zstd_path} (zstd level {ZSTD_LEVEL})")
        _compress_file_zstd(write_path, zstd_path, level=ZSTD_LEVEL)
        _discard_file(Path(str(write_path) + "-lock"))

    return writer.written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert an SDF to an LMDB and a SMILES file without RDKit "
            "sanitization. Line i of the SMILES file (0-based, no header) "
            "is LMDB key i: SMILES<TAB>SDF title."
        ),
    )
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
    parser.add_argument(
        "--output_smi",
        "-s",
        required=True,
        help="Output SMILES path (SMILES<TAB>title, one record per LMDB key)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes (default: CPU count; 1 = in-process)",
    )
    parser.add_argument(
        "--chunk-mb",
        type=float,
        default=DEFAULT_CHUNK_MB,
        help=f"Approximate SDF megabytes per worker task (default: {DEFAULT_CHUNK_MB})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sdf_path):
        raise ValueError("Input SDF file does not exist")

    process_sdf_to_lmdb_smi(
        sdf_path=args.sdf_path,
        output_lmdb=args.output_lmdb,
        output_smi=args.output_smi,
        workers=args.workers,
        chunk_mb=args.chunk_mb,
    )
