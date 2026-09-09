# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Packed hydrogen-stripped mol records for DrugCLIP LMDBs.

On-disk value (little-endian):

    magic (4s)     = b"DCQP"
    n_atoms (u16)
    n_name  (u16)
    token_ids (n_atoms u8)     -- dictionary indices, hydrogens already removed
    coords  (n_atoms * 3 f16)
    name    (n_name bytes, utf-8)

The reader returns the same dict contract as the pickle path:
``{"atoms", "coordinates", "smi"}`` with ``coordinates`` a one-element list
so AffinityMolDataset is unchanged. Select this path when the first LMDB
value starts with PACKED_MAGIC; pickle / zstd-pickle records keep using
LMDBDataset.
"""

from __future__ import annotations

import os
import pickle
import struct

import lmdb
import numpy as np
import zstandard as zstd

PACKED_MAGIC = b"DCQP"
_HEADER = struct.Struct("<4sHH")  # magic, n_atoms, n_name
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()


def _loads_pickle_value(raw):
    if raw is None:
        return None
    if len(raw) >= 4 and raw[:4] == _ZSTD_MAGIC:
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)
    return pickle.loads(raw)


def _symbol_of(dictionary, idx):
    idx = int(idx)
    if hasattr(dictionary, "__getitem__"):
        try:
            return dictionary[idx]
        except Exception:
            pass
    symbols = getattr(dictionary, "symbols", None)
    if symbols is not None:
        return symbols[idx]
    raise KeyError(f"cannot resolve dictionary index {idx}")


def _index_of(dictionary, symbol):
    if hasattr(dictionary, "index"):
        try:
            return int(dictionary.index(symbol))
        except Exception:
            pass
    unk = getattr(dictionary, "unk_index", None)
    if unk is None and hasattr(dictionary, "unk"):
        unk = dictionary.unk()
    if unk is None:
        raise KeyError(f"unknown atom symbol {symbol!r}")
    return int(unk)


def pack_record(atoms, coordinates, smi, dictionary):
    """Pack one molecule. atoms/coords may still include hydrogens."""
    atoms = np.asarray(atoms)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim == 3:
        coordinates = coordinates[0]
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(-1, 3)
    if atoms.shape[0] != coordinates.shape[0]:
        n = min(int(atoms.shape[0]), int(coordinates.shape[0]))
        atoms = atoms[:n]
        coordinates = coordinates[:n]
    mask = atoms != "H"
    atoms = atoms[mask]
    coordinates = coordinates[mask]
    n_atoms = int(atoms.shape[0])
    if n_atoms > 65535:
        raise ValueError(f"n_atoms {n_atoms} exceeds uint16")
    token_ids = np.empty((n_atoms,), dtype=np.uint8)
    for i, sym in enumerate(atoms):
        idx = _index_of(dictionary, str(sym))
        if idx < 0 or idx > 255:
            raise ValueError(f"dictionary index {idx} does not fit in uint8")
        token_ids[i] = idx
    name = str(smi or "").encode("utf-8")
    if len(name) > 65535:
        name = name[:65535]
    coords_f16 = np.ascontiguousarray(coordinates, dtype=np.float16)
    header = _HEADER.pack(PACKED_MAGIC, n_atoms, len(name))
    return header + token_ids.tobytes() + coords_f16.tobytes() + name


def unpack_record(blob, dictionary):
    if blob is None or len(blob) < _HEADER.size:
        raise ValueError("truncated packed record")
    magic, n_atoms, n_name = _HEADER.unpack_from(blob, 0)
    if magic != PACKED_MAGIC:
        raise ValueError(f"bad packed magic {magic!r}")
    off = _HEADER.size
    token_end = off + n_atoms
    coord_end = token_end + n_atoms * 3 * 2
    name_end = coord_end + n_name
    if len(blob) < name_end:
        raise ValueError("truncated packed payload")
    token_ids = np.frombuffer(blob[off:token_end], dtype=np.uint8)
    coords = np.frombuffer(blob[token_end:coord_end], dtype=np.float16).reshape(
        n_atoms, 3
    )
    name = blob[coord_end:name_end].decode("utf-8", errors="replace")
    atoms = np.array([_symbol_of(dictionary, int(i)) for i in token_ids], dtype=object)
    return {
        "atoms": atoms,
        "coordinates": [np.ascontiguousarray(coords, dtype=np.float32)],
        "smi": name,
    }


def is_packed_blob(blob):
    return blob is not None and len(blob) >= 4 and blob[:4] == PACKED_MAGIC


class _NumericLmdb:
    """Read-only numeric-key LMDB ('0'..'N-1'). Independent of unicore."""

    def __init__(self, db_path, readahead=False):
        self.db_path = db_path
        self.readahead = bool(readahead)
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError("{} not found".format(self.db_path))
        env = self._open(readonly=True)
        try:
            with env.begin() as txn:
                n_entries = txn.stat()["entries"]
                first = txn.get(b"0")
                last = txn.get(str(n_entries - 1).encode("ascii")) if n_entries else None
                if n_entries > 0 and first is not None and last is not None:
                    self._keys = None
                    self._length = n_entries
                else:
                    self._keys = list(txn.cursor().iternext(values=False))
                    self._length = len(self._keys)
        finally:
            env.close()
        self.env = None

    def _open(self, readonly=True):
        return lmdb.open(
            self.db_path,
            subdir=False,
            readonly=readonly,
            lock=False,
            readahead=self.readahead,
            meminit=False,
            max_readers=256,
        )

    def __len__(self):
        return self._length

    def close(self):
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            self.env = None

    def get_raw(self, idx):
        if self.env is None:
            self.env = self._open(readonly=True)
        if self._keys is not None:
            key = self._keys[idx]
        else:
            key = f"{idx}".encode("ascii")
        return self.env.begin().get(key)


def lmdb_is_packed(path):
    """True if key 0 (or the first cursor value) is a packed record."""
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=16,
    )
    try:
        with env.begin() as txn:
            raw = txn.get(b"0")
            if raw is None:
                cur = txn.cursor()
                if not cur.first():
                    return False
                raw = cur.value()
        return is_packed_blob(raw)
    finally:
        env.close()


class PickleLMDBDataset:
    """LMDBDataset-compatible reader for pickle / zstd-pickle mol records."""

    def __init__(self, db_path, readahead=False):
        self.db_path = db_path
        self._inner = _NumericLmdb(db_path, readahead=readahead)

    def __len__(self):
        return len(self._inner)

    def close(self):
        self._inner.close()

    def get_raw(self, idx):
        return self._inner.get_raw(idx)

    def __getitem__(self, idx):
        return _loads_pickle_value(self._inner.get_raw(idx))


class PackedLMDBDataset:
    """LMDBDataset-compatible reader that unpacks DCQP records.

    ``dictionary`` maps token ids back to atom symbols so the dict contract
    matches pickle records.
    """

    def __init__(self, db_path, dictionary, readahead=False):
        self.db_path = db_path
        self.dictionary = dictionary
        self._inner = _NumericLmdb(db_path, readahead=readahead)

    def __len__(self):
        return len(self._inner)

    def close(self):
        self._inner.close()

    def get_raw(self, idx):
        return self._inner.get_raw(idx)

    def __getitem__(self, idx):
        raw = self._inner.get_raw(idx)
        return unpack_record(raw, self.dictionary)


def open_mol_lmdb(path, dictionary=None, readahead=False):
    """Open a mol LMDB, auto-selecting packed vs pickle records.

    Packed shards need ``dictionary`` to decode atom token ids. Pickle shards
    ignore it. Both return ``{atoms, coordinates, smi}``.
    """
    if lmdb_is_packed(path):
        if dictionary is None:
            raise ValueError(
                f"packed mol LMDB {path} requires a mol dictionary to decode atom ids"
            )
        return PackedLMDBDataset(path, dictionary, readahead=readahead)
    return PickleLMDBDataset(path, readahead=readahead)


def _record_from_pickle_blob(raw):
    data = _loads_pickle_value(raw)
    atoms = data["atoms"]
    coordinates = data["coordinates"]
    smi = data.get("smi", data.get("smiles", ""))
    return atoms, coordinates, smi


def pack_lmdb_file(src_path, dst_path, dictionary, map_size=None, commit_every=50000):
    """Rewrite src (pickle or packed) as packed LMDB at dst_path.

    Returns n_mols written. Does not delete src.
    """
    src = _NumericLmdb(src_path, readahead=True)
    n = len(src)
    src_size = os.path.getsize(src_path)
    if map_size is None:
        map_size = max(int(src_size * 0.8) + 64 * 1024 * 1024, 256 * 1024 * 1024)

    dst_path = str(dst_path)
    parent = os.path.dirname(os.path.abspath(dst_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(dst_path):
        os.remove(dst_path)

    env_out = lmdb.open(
        dst_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=int(map_size),
    )
    txn = env_out.begin(write=True)
    written = 0
    try:
        src.env = src._open(readonly=True)
        with src.env.begin() as txn_in:
            for i in range(n):
                key = f"{i}".encode("ascii") if src._keys is None else src._keys[i]
                raw = txn_in.get(key)
                if raw is None:
                    raise KeyError(f"missing LMDB key for index {i} in {src_path}")
                if is_packed_blob(raw):
                    blob = raw
                else:
                    atoms, coordinates, smi = _record_from_pickle_blob(raw)
                    blob = pack_record(atoms, coordinates, smi, dictionary)
                txn.put(f"{i}".encode("ascii"), blob)
                written += 1
                if written % commit_every == 0:
                    txn.commit()
                    txn = env_out.begin(write=True)
        txn.commit()
    except lmdb.MapFullError:
        try:
            txn.abort()
        except Exception:
            pass
        env_out.close()
        src.close()
        return pack_lmdb_file(
            src_path,
            dst_path,
            dictionary,
            map_size=int(map_size) * 2,
            commit_every=commit_every,
        )
    except Exception:
        try:
            txn.abort()
        except Exception:
            pass
        env_out.close()
        src.close()
        raise
    env_out.sync()
    env_out.close()
    src.close()
    return written
