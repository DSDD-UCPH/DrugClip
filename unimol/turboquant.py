# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""MSE-only TurboQuant for L2-normalized DrugCLIP embeddings.

Random rotation + Lloyd-Max scalar quantization of sphere coordinates,
using the exact d-dimensional Beta coordinate law (not the large-d Gaussian
limit). Packed codes: 1-bit 8 coords/byte, 2-bit 4 coords/byte, 3-bit 8
coords/3 bytes, 4-bit 2 coords/byte. There is no QJL residual
path: ranking z-scores cancel a global multiplicative bias, so extra bits
spent on unbiased inner products would hurt MSE at the same budget.

Codes store already-rotated coordinates. Scoring is
`rotate(query) @ dequant(codes).T`.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

TQ_VERSION = "tq-mse-d128-b4-v1"
DEFAULT_DIM = 128
DEFAULT_BITS = 4


def random_rotation(dim, seed=1):
    """Haar-ish orthogonal matrix via QR of a Gaussian."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal((dim, dim), dtype=np.float64)
    q, r = np.linalg.qr(g)
    # Uniformize signs so Q is Haar-distributed on O(d).
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    # Det-1 (SO(d)): flip one column if needed. Harmless for MSE.
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(np.float32)


def sample_sphere_coords(dim, n_samples, seed=0):
    """Draw coordinate 0 of uniform points on the unit sphere S^{d-1}."""
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((int(n_samples), int(dim)), dtype=np.float64)
    nrm = np.linalg.norm(z, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return (z[:, 0] / nrm[:, 0]).astype(np.float64)


def fit_lloyd_max(samples, n_levels, n_iter=80):
    """1-D Lloyd-Max (k-means) on scalar samples in [-1, 1]."""
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    n_levels = int(n_levels)
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    qs = (np.arange(n_levels, dtype=np.float64) + 0.5) / n_levels
    centroids = np.quantile(samples, qs)
    centroids = np.clip(centroids, -1.0, 1.0)
    for _ in range(int(n_iter)):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        labels = np.searchsorted(boundaries, samples, side="right")
        for k in range(n_levels):
            sel = labels == k
            if np.any(sel):
                centroids[k] = samples[sel].mean()
        if np.any(np.diff(centroids) <= 0):
            centroids = np.sort(centroids)
            # Nudge collisions.
            for k in range(1, n_levels):
                if centroids[k] <= centroids[k - 1]:
                    centroids[k] = centroids[k - 1] + 1e-8
    return centroids.astype(np.float32)


def packed_code_width(dim, bits):
    """Bytes per vector for packed codes of `bits` bits and dimension `dim`."""
    dim = int(dim)
    bits = int(bits)
    if bits in (1, 2, 4):
        group = 8 // bits
        if dim % group != 0:
            raise ValueError(
                f"dim={dim} not divisible by {group} for {bits}-bit packing"
            )
        return dim // group
    if bits == 3:
        if dim % 8 != 0:
            raise ValueError(f"dim={dim} must be a multiple of 8 for 3-bit packing")
        return (dim // 8) * 3
    raise ValueError(f"no packer for bits={bits}")


def pack_indices(indices, bits):
    """Pack (N, d) uint8 indices into (N, packed_code_width) uint8."""
    idx = np.asarray(indices, dtype=np.uint8)
    if idx.ndim != 2:
        raise ValueError(f"expected (N, d) indices, got {idx.shape}")
    n, dim = idx.shape
    bits = int(bits)
    mask = np.uint8((1 << bits) - 1)
    if bits in (1, 2, 4):
        group = 8 // bits
        width = packed_code_width(dim, bits)
        blocks = (idx.reshape(n, width, group) & mask).astype(np.uint8, copy=False)
        out = np.zeros((n, width), dtype=np.uint8)
        for i in range(group):
            shift = bits * (group - 1 - i)
            out |= np.left_shift(blocks[:, :, i], shift)
        return out
    if bits == 3:
        width = packed_code_width(dim, bits)
        n_groups = dim // 8
        g = (idx.reshape(n, n_groups, 8) & np.uint8(7)).astype(np.uint8, copy=False)
        b0 = np.bitwise_or(
            np.bitwise_or(np.left_shift(g[:, :, 0], 5), np.left_shift(g[:, :, 1], 2)),
            np.right_shift(g[:, :, 2], 1),
        )
        b1 = np.bitwise_or(
            np.bitwise_or(
                np.bitwise_or(
                    np.left_shift(np.bitwise_and(g[:, :, 2], np.uint8(1)), 7),
                    np.left_shift(g[:, :, 3], 4),
                ),
                np.left_shift(g[:, :, 4], 1),
            ),
            np.right_shift(g[:, :, 5], 2),
        )
        b2 = np.bitwise_or(
            np.bitwise_or(
                np.left_shift(np.bitwise_and(g[:, :, 5], np.uint8(3)), 6),
                np.left_shift(g[:, :, 6], 3),
            ),
            g[:, :, 7],
        )
        packed = np.stack((b0, b1, b2), axis=-1).reshape(n, width)
        return packed.astype(np.uint8, copy=False)
    raise ValueError(f"no packer for bits={bits}")


def unpack_indices(codes, dim, bits):
    """Unpack packed uint8 codes to (N, dim) uint8 indices."""
    codes = np.asarray(codes, dtype=np.uint8)
    dim = int(dim)
    bits = int(bits)
    width = packed_code_width(dim, bits)
    if codes.ndim != 2 or codes.shape[1] != width:
        raise ValueError(
            f"codes shape {codes.shape} incompatible with dim={dim} bits={bits} "
            f"(expected width {width})"
        )
    n = codes.shape[0]
    if bits in (1, 2, 4):
        group = 8 // bits
        mask = np.uint8((1 << bits) - 1)
        out = np.empty((n, width, group), dtype=np.uint8)
        for i in range(group):
            shift = bits * (group - 1 - i)
            out[:, :, i] = np.bitwise_and(np.right_shift(codes, shift), mask)
        return out.reshape(n, dim)
    if bits == 3:
        n_groups = dim // 8
        c = codes.reshape(n, n_groups, 3)
        b0 = c[:, :, 0]
        b1 = c[:, :, 1]
        b2 = c[:, :, 2]
        g = np.empty((n, n_groups, 8), dtype=np.uint8)
        g[:, :, 0] = np.right_shift(b0, 5)
        g[:, :, 1] = np.bitwise_and(np.right_shift(b0, 2), np.uint8(7))
        g[:, :, 2] = np.bitwise_or(
            np.left_shift(np.bitwise_and(b0, np.uint8(3)), 1),
            np.right_shift(b1, 7),
        )
        g[:, :, 3] = np.bitwise_and(np.right_shift(b1, 4), np.uint8(7))
        g[:, :, 4] = np.bitwise_and(np.right_shift(b1, 1), np.uint8(7))
        g[:, :, 5] = np.bitwise_or(
            np.left_shift(np.bitwise_and(b1, np.uint8(1)), 2),
            np.right_shift(b2, 6),
        )
        g[:, :, 6] = np.bitwise_and(np.right_shift(b2, 3), np.uint8(7))
        g[:, :, 7] = np.bitwise_and(b2, np.uint8(7))
        return g.reshape(n, dim)
    raise ValueError(f"no packer for bits={bits}")


def pack_indices_4bit(indices):
    """Pack (N, d) uint8 indices with d even into (N, d/2) uint8 (hi nibble first)."""
    return pack_indices(indices, 4)


def unpack_indices_4bit(codes, dim):
    """Unpack (N, dim/2) uint8 to (N, dim) uint8 indices."""
    return unpack_indices(codes, dim, 4)


def codec_version(dim=DEFAULT_DIM, bits=DEFAULT_BITS):
    return f"tq-mse-d{int(dim)}-b{int(bits)}-v1"


class TurboQuant:
    """Frozen rotation + scalar codebook. Codes are packed 1/2/3/4-bit.

    1-bit: 8 coords/byte; 2-bit: 4 coords/byte; 3-bit: 8 coords/3 bytes;
    4-bit: 2 coords/byte (hi nibble first). Scoring is
    `rotate(query) @ dequant(codes).T`.
    """

    def __init__(self, rotation, codebook, version=TQ_VERSION, bits=DEFAULT_BITS):
        self.rotation = np.ascontiguousarray(rotation, dtype=np.float32)
        self.codebook = np.ascontiguousarray(codebook, dtype=np.float32).reshape(-1)
        self.version = str(version)
        self.bits = int(bits)
        self.dim = int(self.rotation.shape[0])
        if self.rotation.shape != (self.dim, self.dim):
            raise ValueError("rotation must be square")
        if self.bits < 1 or self.bits > 8:
            raise ValueError(f"bits must be in 1..8, got {self.bits}")
        n_levels = 1 << self.bits
        if self.codebook.shape[0] != n_levels:
            raise ValueError(
                f"codebook length {self.codebook.shape[0]} != {n_levels} for bits={self.bits}"
            )
        self.n_levels = n_levels
        self.packed = self.bits in (1, 2, 3, 4)
        if not self.packed:
            self.code_width = self.dim
        else:
            self.code_width = packed_code_width(self.dim, self.bits)

    @classmethod
    def create(
        cls,
        dim=DEFAULT_DIM,
        bits=DEFAULT_BITS,
        seed=1,
        n_samples=2_000_000,
        n_iter=80,
        version=TQ_VERSION,
    ):
        rotation = random_rotation(dim, seed=seed)
        samples = sample_sphere_coords(dim, n_samples, seed=seed + 1)
        codebook = fit_lloyd_max(samples, 1 << int(bits), n_iter=n_iter)
        if version == TQ_VERSION and int(bits) != DEFAULT_BITS:
            version = codec_version(dim, bits)
        return cls(rotation, codebook, version=version, bits=bits)

    def save(self, path):
        np.savez(
            path,
            rotation=self.rotation,
            codebook=self.codebook,
            version=np.array(self.version),
            bits=np.int32(self.bits),
            dim=np.int32(self.dim),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        version = data["version"]
        if version.shape == ():
            version = str(version)
        else:
            version = str(version.item() if hasattr(version, "item") else version)
        bits = int(data["bits"]) if "bits" in data.files else DEFAULT_BITS
        return cls(data["rotation"], data["codebook"], version=version, bits=bits)

    def rotate(self, x):
        """x is (N, d) in the original embedding frame -> rotated frame."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        return x @ self.rotation

    def rotate_query(self, x):
        """Same as rotate; named for the scan path (query stays in original frame)."""
        return self.rotate(x)

    def quantize_indices(self, x, already_rotated=False):
        """Nearest codebook index per coordinate. x: (N, d) unit vectors."""
        y = np.ascontiguousarray(x, dtype=np.float32)
        if not already_rotated:
            y = self.rotate(y)
        # (N, d, 1) - (L,) -> (N, d, L)
        dist = np.abs(y[..., None] - self.codebook.reshape(1, 1, -1))
        return dist.argmin(axis=-1).astype(np.uint8)

    def quantize(self, x, already_rotated=False):
        """Packed (N, code_width) uint8 codes, or unpacked (N, d) if bits>4."""
        idx = self.quantize_indices(x, already_rotated=already_rotated)
        if self.packed:
            return pack_indices(idx, self.bits)
        return idx

    def dequantize(self, codes, packed=None):
        """codes packed (N, code_width) uint8 or unpacked (N, d) indices -> (N, d) float32 in rotated frame."""
        if packed is None:
            packed = self.packed
        if packed:
            idx = unpack_indices(codes, self.dim, self.bits)
        else:
            idx = np.asarray(codes, dtype=np.uint8)
        return self.codebook[idx]

    def reconstruct(self, codes, packed=None):
        """Dequantize and unrotate into the original embedding frame."""
        y = self.dequantize(codes, packed=packed)
        return y @ self.rotation.T

    def metadata(self):
        return {
            "version": self.version,
            "dim": self.dim,
            "bits": self.bits,
            "n_levels": self.n_levels,
            "code_width": self.code_width,
        }


def write_codes_chunked(path, embs, tq, chunk=65536):
    """Quantize embeddings in row chunks into a uint8 memmap at path.

    Avoids the (N, d, n_levels) temporary that a single tq.quantize(embs) would
    allocate. Packed widths: 1-bit dim/8, 2-bit dim/4, 3-bit 3*dim/8, 4-bit dim/2.
    """
    n = int(embs.shape[0])
    width = int(tq.code_width)
    tmp = path + ".tmp.npy"
    if os.path.exists(tmp):
        os.remove(tmp)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    mm = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(n, width)
    )
    chunk = max(1, int(chunk))
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        mm[start:end] = tq.quantize(embs[start:end])
    mm.flush()
    del mm
    os.replace(tmp, path)
    return path


def dequant_gemm_numpy(query_orig, codes, tq, chunk_mols=1_048_576):
    """Score (n_query, n_mols) = rotate(query) @ dequant(codes).T, chunked.

    query_orig: (Q, d) float32 original-frame embeddings (L2-normalized).
    codes: (N, code_width) uint8 mmap-friendly packed codes.
    """
    q_rot = tq.rotate_query(np.ascontiguousarray(query_orig, dtype=np.float32))
    n_mols = int(codes.shape[0])
    n_q = int(q_rot.shape[0])
    out = np.empty((n_q, n_mols), dtype=np.float32)
    for start in range(0, n_mols, int(chunk_mols)):
        end = min(n_mols, start + int(chunk_mols))
        y = tq.dequantize(codes[start:end], packed=True)
        out[:, start:end] = q_rot @ y.T
    return out


def dequant_gemm_torch(query_orig, codes, tq, chunk_mols=524288, device=None):
    """GPU chunked GEMM. Returns a CPU float32 (Q, N) array.

    Falls back to numpy if torch or CUDA is unavailable.
    """
    try:
        import torch
    except ImportError:
        return dequant_gemm_numpy(query_orig, codes, tq, chunk_mols=chunk_mols)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook = torch.from_numpy(np.ascontiguousarray(tq.codebook)).to(device)
    rot = torch.from_numpy(np.ascontiguousarray(tq.rotation)).to(device)
    q = torch.from_numpy(np.ascontiguousarray(query_orig, dtype=np.float32)).to(device)
    q_rot = q @ rot
    n_mols = int(codes.shape[0])
    n_q = int(q_rot.shape[0])
    out = np.empty((n_q, n_mols), dtype=np.float32)
    dim = tq.dim
    bits = int(tq.bits)
    for start in range(0, n_mols, int(chunk_mols)):
        end = min(n_mols, start + int(chunk_mols))
        packed = np.array(codes[start:end], dtype=np.uint8, copy=True)
        if bits == 4:
            t = torch.from_numpy(packed).to(device, non_blocking=True)
            hi = torch.bitwise_right_shift(t, 4).to(torch.long)
            lo = torch.bitwise_and(t, 0x0F).to(torch.long)
            idx = torch.stack((hi, lo), dim=-1).reshape(t.shape[0], dim)
        else:
            idx_np = unpack_indices(packed, dim, bits)
            idx = torch.from_numpy(np.ascontiguousarray(idx_np)).to(
                device, dtype=torch.long, non_blocking=True
            )
        y = codebook[idx]  # (B, dim)
        block = q_rot @ y.t()
        out[:, start:end] = block.detach().float().cpu().numpy()
    return out


def quantize_prod_control(x, tq_low_bits, bits=4, seed=1):
    """Cheap inner-product control: (bits-1) MSE + 1-bit QJL on the residual.

    Returns reconstructed original-frame vectors. Used only by the Phase 0
    quant-delta diagnostic; not part of the production codec.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = tq_low_bits.rotate(x)
    idx = tq_low_bits.quantize_indices(y, already_rotated=True)
    y_mse = tq_low_bits.codebook[idx]
    x_mse = y_mse @ tq_low_bits.rotation.T
    r = x - x_mse
    gamma = np.linalg.norm(r, axis=1, keepdims=True).astype(np.float32)
    gamma = np.maximum(gamma, 1e-8)
    rng = np.random.default_rng(int(seed))
    d = x.shape[1]
    s = rng.standard_normal((d, d), dtype=np.float32)
    sr = (r / gamma) @ s.T
    signs = np.sign(sr)
    signs[signs == 0] = 1.0
    # E[s_i^T r * sign(s_i^T r)] reconstruction from the paper.
    x_qjl = (np.sqrt(np.pi / 2.0) / d) * (signs @ s) * gamma
    return x_mse + x_qjl.astype(np.float32)


def mse_reconstruct_unpacked(x, rotation, codebook):
    """Rotate, nearest-centroid per coord, unrotate. Used for 3-bit Phase 0 controls.

    Returns (x_hat, indices) with indices uint8 of shape (N, d).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    rotation = np.ascontiguousarray(rotation, dtype=np.float32)
    codebook = np.ascontiguousarray(codebook, dtype=np.float32).reshape(-1)
    y = x @ rotation
    dist = np.abs(y[..., None] - codebook.reshape(1, 1, -1))
    idx = dist.argmin(axis=-1).astype(np.uint8)
    y_hat = codebook[idx]
    return y_hat @ rotation.T, idx


def dumps_meta(tq):
    return json.dumps(tq.metadata(), sort_keys=True)
