# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
import time
from functools import lru_cache
import logging
import numpy as np
import zstandard as zstd

logger = logging.getLogger(__name__)

# zstd frame magic; used to detect compressed LMDB values from sdf_to_lmdb.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()


def _loads_lmdb_value(raw):
    """Unpickle an LMDB value, decompressing zstd frames when present."""
    if raw is None:
        return None
    if len(raw) >= 4 and raw[:4] == _ZSTD_MAGIC:
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)
    return pickle.loads(raw)


class LMDBDataset:
    def __init__(self, db_path, readahead=False):
        self.db_path = db_path
        self.readahead = bool(readahead)
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        try:
            with env.begin() as txn:
                # Prefer O(1) entry count for numeric-key libs ("0".."N-1"). Full key
                # walks are minutes on 10M+ HDD LMDBs and are only needed for __len__.
                n_entries = txn.stat()["entries"]
                if n_entries > 0 and self._is_numeric_key_layout(txn, n_entries):
                    self._keys = None
                    self._length = n_entries
                else:
                    self._keys = list(txn.cursor().iternext(values=False))
                    self._length = len(self._keys)
        finally:
            env.close()

    @staticmethod
    def _is_numeric_key_layout(txn, n_entries):
        first = txn.get(b"0")
        last = txn.get(str(n_entries - 1).encode("ascii"))
        return first is not None and last is not None

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=self.readahead,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return self._length

    def close(self):
        # Release the process-local LMDB env (LMDB allows only one open handle
        # per path per process). Safe to call multiple times.
        env = getattr(self, "env", None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            try:
                del self.env
            except AttributeError:
                pass
        try:
            self.__getitem__.cache_clear()
        except Exception:
            pass

    def get_raw(self, idx):
        # Return the stored value bytes without unpickling (for byte-copy compaction).
        # May be raw pickle or zstd-framed pickle; callers that need a dict should
        # use __getitem__ / _loads_lmdb_value instead.
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        if self._keys is not None:
            key = self._keys[idx]
        else:
            key = f"{idx}".encode("ascii")
        return self.env.begin().get(key)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return _loads_lmdb_value(self.get_raw(idx))


def compact_lmdb_indices(src_path, indices, dst_path, src_readahead=True):
    """Copy selected LMDB entries into a new sequential LMDB (keys \"0\"..).

    Values are copied as raw pickled blobs (no unpickle/re-pickle). `indices`
    are source integer indices into a numeric-key library, or positions into
    the key list for non-numeric layouts.
    """
    t0 = time.perf_counter()
    src = LMDBDataset(src_path, readahead=src_readahead)
    src.connect_db(src.db_path, save_to_self=True)
    n_src = len(src)
    n_out = len(indices)
    src_size = os.path.getsize(src_path)
    frac = (n_out / n_src) if n_src > 0 else 1.0
    # Size for the survivor subset plus headroom; LMDB map_size is an upper bound.
    map_size = max(int(src_size * frac * 1.5) + 64 * 1024 * 1024, 256 * 1024 * 1024)
    bytes_written = 0

    def _open_out(size):
        return lmdb.open(
            dst_path,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            map_size=size,
        )

    def _key_for(src_i):
        src_i = int(src_i)
        if src._keys is not None:
            return src._keys[src_i]
        return f"{src_i}".encode("ascii")

    env_out = _open_out(map_size)
    # Larger commits: fewer fsyncs on multi-million survivor copies.
    commit_every = 50000
    # First index not yet committed; on MapFullError resume here (no full replay).
    start_i = 0
    txn_out = env_out.begin(write=True)
    try:
        with src.env.begin() as txn_in:
            while start_i < n_out:
                try:
                    for out_i in range(start_i, n_out):
                        raw = txn_in.get(_key_for(indices[out_i]))
                        if raw is None:
                            raise KeyError(
                                f"missing LMDB key for index {indices[out_i]} in {src_path}"
                            )
                        txn_out.put(f"{out_i}".encode("ascii"), raw)
                        bytes_written += len(raw)
                        if (out_i + 1) % commit_every == 0:
                            txn_out.commit()
                            start_i = out_i + 1
                            txn_out = env_out.begin(write=True)
                    txn_out.commit()
                    start_i = n_out
                except lmdb.MapFullError as e:
                    try:
                        txn_out.abort()
                    except Exception:
                        pass
                    env_out.close()
                    # Estimate remaining bytes from the failed raw if available.
                    map_size = max(map_size * 2, map_size + 256 * 1024 * 1024)
                    logger.warning(
                        f"compact LMDB map full at {start_i}/{n_out}; "
                        f"growing map_size to {map_size} and resuming "
                        f"(dst={dst_path})"
                    )
                    env_out = _open_out(map_size)
                    txn_out = env_out.begin(write=True)
                    if start_i >= n_out:
                        raise e
    except Exception:
        try:
            txn_out.abort()
        except Exception:
            pass
        env_out.close()
        raise
    env_out.sync()
    env_out.close()
    elapsed = time.perf_counter() - t0
    mols_per_sec = n_out / elapsed if elapsed > 0 else 0.0
    mb = bytes_written / (1024 * 1024)
    logger.info(
        f"compact_lmdb_indices: wrote {n_out} mols ({mb:.1f} MiB raw) to {dst_path} "
        f"in {elapsed:.2f}s ({mols_per_sec:.0f} mols/s, map_size={map_size})"
    )
    return n_out


def resolve_mol_smiles(db_path, indices, readahead=False):
    """Return SMILES strings for integer LMDB indices (one read transaction)."""
    ds = LMDBDataset(db_path, readahead=readahead)
    ds.connect_db(ds.db_path, save_to_self=True)
    out = []
    with ds.env.begin() as txn:
        for idx in indices:
            idx = int(idx)
            if ds._keys is not None:
                key = ds._keys[idx]
            else:
                key = f"{idx}".encode("ascii")
            raw = txn.get(key)
            if raw is None:
                out.append("")
                continue
            data = _loads_lmdb_value(raw)
            if "smiles" in data:
                out.append(data["smiles"])
            elif "smi" in data:
                out.append(data["smi"])
            else:
                out.append("")
    ds.close()
    return out


import logging
import pickle as pkl
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Union

import lmdb
import zstandard as zstd

logger = logging.getLogger(__name__)

# 1T map_size, ref: https://lmdb.readthedocs.io/en/release/#environment-class
MAP_SIZE = 10 * 1024 * 1024 * 1024 * 1024  # 10T


class LMDBDatasetV2:
    """
    split:
        full: "key1,key2,..."
        dataset1: "key1,key2,..."
        dataset2: "key2,key3,..."
        train: "key1,key2,..."
        val: "key3,key4,..."
    data:
        key1: pkl.dump({k1: v11, k2: v12, ...})
        key2: pkl.dump({k1: v21, k2: v22, ...})
    """

    SPLIT_DB = "split"
    DATA_DB = "data"
    MAX_CACHE_SIZE = 16
    MAX_WRITE_RETRY_WHEN_LOCK = 10
    RETRY_INTERVAL = 5

    def __init__(
        self,
        lmdb_path: Union[str, Path],
        compressed: bool = True,
        readonly: bool = True,
        enable_cache: bool = True,
    ):
        self.lmdb_path = Path(lmdb_path).resolve()
        self.compressed = compressed
        if self.compressed:
            self.compressor = zstd.ZstdCompressor(level=3)
            self.decompressor = zstd.ZstdDecompressor()
        self.readonly = readonly
        self.enable_cache = enable_cache

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                self.env = lmdb.open(
                    str(self.lmdb_path),
                    max_dbs=2,
                    map_size=MAP_SIZE,
                    readonly=readonly,
                    lock=not readonly,
                    create=not readonly,
                )
                break
            except Exception as e:
                time.sleep(self.RETRY_INTERVAL)
        else:
            raise Exception(
                f"Failed to open lmdb after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
            )
        self.db = {
            db_name: self.env.open_db(db_name.encode())
            for db_name in [self.SPLIT_DB, self.DATA_DB]
        }
        self._splits = dict()
        self.default_split = "full"

    def compress(self, content: bytes) -> bytes:
        content = self.compressor.compress(content)
        return content

    def decompress(self, content: bytes) -> bytes:
        content = self.decompressor.decompress(content)
        return content

    def close(self):
        self.env.close()

    def _get_value(self, db: str, key: str, default: Any = None) -> Any:
        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=False) as txn:
                    value = txn.get(key.encode())
                break
            except Exception as e:
                time.sleep(self.RETRY_INTERVAL)
        else:
            raise Exception(
                f"Failed to read after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
            )

        if value is not None:
            if self.compressed:
                value = self.decompress(value)
        else:
            logger.warning(f"Key {key} not found in {db}")
            value = default
        return value

    def _set_value(self, db: str, key: str, value: bytes):
        if self.compressed:
            value = self.compress(value)

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=True) as txn:
                    txn.put(key.encode(), value)
                return
            except Exception as e:
                time.sleep(self.RETRY_INTERVAL)
        raise Exception(
            f"Failed to write after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        )

    def _set_values(self, db: str, data: Dict[str, Any]):
        """better for data construction with large data size."""
        compressed = []
        for key, value in data.items():
            if self.compressed:
                value = self.compress(value)
            compressed.append((key, value))

        for _ in range(self.MAX_WRITE_RETRY_WHEN_LOCK):
            try:
                with self.env.begin(db=self.db[db], write=True) as txn:
                    for key, value in compressed:
                        txn.put(key.encode(), value)
                return
            except Exception as e:
                time.sleep(self.RETRY_INTERVAL)
        raise Exception(
            f"Failed to write after {self.MAX_WRITE_RETRY_WHEN_LOCK} retries"
        )

    def _smart_decode_list(self, value: bytes) -> List[Any]:
        try:
            return value.decode().split(",")
        except Exception:
            return pkl.loads(value)

    def _smart_encode_list(self, value: List[Any]) -> bytes:
        if len(value) == 0:
            return pkl.dumps([])
        elif type(value[0]) == str:
            return ",".join(value).encode()
        else:
            return pkl.dumps(value)

    def get_split(self, key: str) -> List[str]:
        if key not in self._splits or not self.enable_cache:
            split_save = self._get_value(self.SPLIT_DB, key)
            if split_save is None:
                return []
            else:
                self._splits[key] = self._smart_decode_list(split_save)
        return self._splits[key]

    def set_split(
        self,
        split: str,
        keys: List[str],
        append: bool = False,
        deduplicate: bool = True,
        update_full: bool = False,
        temporary: bool = False,
    ):
        if append:
            keys = self.get_split(split) + keys

        if deduplicate:
            keys = list(sorted(list(set(keys))))

        if not temporary:
            self._set_value(self.SPLIT_DB, split, self._smart_encode_list(keys))
        self._splits[split] = keys
        if (
            split != "full"
            and update_full
            and len(keys) > 0
            and type(keys[0]) == str
        ):
            self.update_full_split()

    def update_full_split(self, from_data: bool = False):
        keys = []
        if from_data:
            with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
                keys = [
                    key.decode() for key in txn.cursor().iternext(values=False)
                ]
        else:  # from split
            with self.env.begin(db=self.db[self.SPLIT_DB], write=False) as txn:
                splits = [
                    key.decode() for key in txn.cursor().iternext(values=False)
                ]
            for split in splits:
                if split != "full":
                    keys.extend(self.get_split(split))
                    keys = list(sorted(list(set(keys))))
        if not self.readonly:
            self.set_split("full", keys)
        else:
            self._splits["full"] = keys

    def check_keys(self):
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        full_keys = self.get_split("full")

        missing_keys = list(set(full_keys) - set(keys))
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")

        orphan_keys = list(set(keys) - set(full_keys))
        if len(orphan_keys) > 0:
            print(f"Orphan keys: {orphan_keys}")

        if len(missing_keys) + len(orphan_keys) == 0:
            print("All keys are in place")

    def __getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if self.enable_cache:
            return self.__cache_getitem__(key)
        else:
            return self.__imp_getitem__(key)

    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def __cache_getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        return self.__imp_getitem__(key)

    def __imp_getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if type(key) == int:
            key = self.get_split(self.default_split)[key]
        data = self._get_value(self.DATA_DB, key)
        if data is not None:
            data = pkl.loads(data)
        return data

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self._set_value(self.DATA_DB, key, pkl.dumps(value))

    def write_data(self, data: Dict[str, Dict[str, Any]]) -> None:
        """when writing large amount of data, use `write_data` to call
        _set_values instead of __setitem__"""
        self._set_values(
            self.DATA_DB, {key: pkl.dumps(value) for key, value in data.items()}
        )

    def __contains__(self, key: str) -> bool:
        with self.env.begin(db=self.db[self.DATA_DB], write=False) as txn:
            return txn.get(key.encode()) is not None

    def set_default_split(self, split: str) -> None:
        self.default_split = split
        if self.enable_cache:
            self.__cache_getitem__.cache_clear()

    def __len__(self) -> int:
        keys = self.get_split(self.default_split)
        return len(keys)

    def summary(self) -> Dict[str, int]:
        self.update_full_split()
        return {split: len(keys) for split, keys in self._splits.items()}

    def __repr__(self) -> str:
        return f"LMDBDataset({self.lmdb_path})"

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self):
            result = self[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration


class LMDBKeyDataset(LMDBDatasetV2):
    def __init__(
        self, 
        lmdb_path,         
        compressed: bool = True,
        readonly: bool = True,
        enable_cache: bool = True,
    ):
        super().__init__(lmdb_path, compressed = compressed, readonly = readonly, enable_cache = enable_cache)
    
    def __getitem__(self, key: Union[str, int]) -> Dict[str, Any]:
        if isinstance(key, int) or isinstance(key, np.int64):
            key = self.get_split(self.default_split)[key]
        return key