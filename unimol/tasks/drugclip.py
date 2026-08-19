# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import gc
import hashlib
import logging
import os
import shutil
import tempfile
import time
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord, LMDBDatasetV2,
                         AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
from unimol.data.lmdb_dataset import (
    LMDBDataset as MolLMDBDataset,
    compact_lmdb_indices,
)
from unimol.tasks._drugclip_rank import (
    _SCORE_MEMMAP_SELECT_CHUNK,
    _score_memmap_dtype,
    _cascade_score_memmap_dtype,
    _cascade_anchor_sample_size,
    _cascade_anchor_exact,
    _ceil_to_batch,
    _cascade_tier_bsz,
    _cascade_score_bytes,
    _cascade_scores_need_memmap,
    _mean_fold_scores,
    _ensemble_rank_mols,
    _AnchorReservoir,
    _robust_pocket_anchors,
    _robust_pocket_anchors_memmap,
    _dense_score_chunk_size,
    _max_zscore_metric,
    _batch_gate_metric,
    _max_zscore_metric_from_folds,
    _gather_score_columns,
    _cascade_tier0_select_gather,
    _log_cascade_scratch,
)
import h5py


logger = logging.getLogger(__name__)

# Fraction of the molecule library written to retrieval output (full, cascade, and
# cascade recall diagnostics). Change this single value to adjust the top-k cut.
RETRIEVAL_TOP_FRAC = 0.01

# Default DataLoader batch sizes when --retrieval-bsz is 0 / unset.
_DEFAULT_RETRIEVAL_BSZ_FULL = 384
_DEFAULT_RETRIEVAL_BSZ_CASCADE = 256

# Suffix for on-disk mol lib_cache dirs. Embeddings follow LMDB order as
# provided. Bump when the cache layout changes.
_MOL_LIB_CACHE_VERSION = "lmdbsort"

# Weight / pickle layout for ensemble fold_version strings.
_FOLD_VERSION_SPECS = {
    "6_folds": ("6_folds", 6),
    "8_folds": ("8_folds", 8),
    "6_folds_filtered": ("6_folds", 6),
}



@register_task("drugclip")
class DrugCLIP(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # Keep [MASK] in the vocab for checkpoint size alignment.
        dictionary.add_symbol("[MASK]", is_special=True)
        pocket_dictionary.add_symbol("[MASK]", is_special=True)

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    @staticmethod
    def _prepend_and_append(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    @staticmethod
    def _fold_checkpoints(fold_version):
        if fold_version not in _FOLD_VERSION_SPECS:
            raise ValueError(f"unknown fold_version: {fold_version}")
        weight_dir, n_folds = _FOLD_VERSION_SPECS[fold_version]
        return [
            f"./data/model_weights/{weight_dir}/fold_{i}.pt" for i in range(n_folds)
        ]

    @staticmethod
    def _mol_cache_paths(fold_version):
        if fold_version not in _FOLD_VERSION_SPECS:
            raise ValueError(f"unknown fold_version: {fold_version}")
        _, n_folds = _FOLD_VERSION_SPECS[fold_version]
        return [
            f"./data/encoded_mol_embs/{fold_version}/fold{i}.pkl"
            for i in range(n_folds)
        ]

    def _mol_graph_from_apo(self, apo_dataset):
        # Tokenize / distance nest pieces from a dataset with normalized
        # atoms/coordinates. Returns (net_input, mol_len, src_dataset).
        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = self._prepend_and_append(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = self._prepend_and_append(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        net_input = {
            "mol_src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "mol_src_distance": RightPadDataset2D(
                distance_dataset,
                pad_idx=0,
            ),
            "mol_src_edge_type": RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),
        }
        return net_input, RawArrayDataset(mol_len_dataset), src_dataset

    def _pocket_graph_from_apo(self, apo_dataset, include_coord=True):
        # Tokenize / distance nest pieces from a dataset with normalized
        # pocket_atoms/pocket_coordinates.
        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = self._prepend_and_append(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = self._prepend_and_append(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )
        net_input = {
            "pocket_src_tokens": RightPadDataset(
                src_pocket_dataset,
                pad_idx=self.pocket_dictionary.pad(),
            ),
            "pocket_src_distance": RightPadDataset2D(
                distance_pocket_dataset,
                pad_idx=0,
            ),
            "pocket_src_edge_type": RightPadDataset2D(
                pocket_edge_type,
                pad_idx=0,
            ),
        }
        if include_coord:
            net_input["pocket_src_coord"] = RightPadDatasetCoord(
                coord_pocket_dataset,
                pad_idx=0,
            )
        return net_input, RawArrayDataset(pocket_len_dataset)

    def _build_mol_nest(self, dataset, extra_fields=None):
        # AffinityMolDataset -> NestedDictionaryDataset with mol net_input.
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)
        apo_dataset = NormalizeDataset(dataset, "coordinates")
        net_input, mol_len, _ = self._mol_graph_from_apo(apo_dataset)
        nest = {"net_input": net_input, "mol_len": mol_len}
        if extra_fields:
            nest.update(extra_fields)
        return NestedDictionaryDataset(nest)

    def _prepare_pocket_apo(self, dataset):
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )
        return dataset

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
        else:
            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

        dataset = self._prepare_pocket_apo(dataset)
        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)
        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        mol_net, mol_len, src_dataset = self._mol_graph_from_apo(apo_dataset)
        pocket_net, pocket_len = self._pocket_graph_from_apo(apo_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    **mol_net,
                    **pocket_net,
                    "mol_len": mol_len,
                    "pocket_len": pocket_len,
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))
            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset

    def load_mols_dataset(self, data_path, atoms, coords, readahead=False, **kwargs):
        # Use the local LMDBDataset (fast open + optional readahead). Sequential
        # retrieval scans should pass readahead=True for HDD-friendly I/O.
        dataset = MolLMDBDataset(data_path, readahead=readahead)
        label_dataset = KeyDataset(dataset, "label", default=0)
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        smi_dataset = KeyDataset(dataset, "smi")
        nest_dataset = self._build_mol_nest(
            dataset,
            extra_fields={
                "smi_name": RawArrayDataset(smi_dataset),
                "target": RawArrayDataset(label_dataset),
            },
        )
        logger.info(f"mol library: {len(nest_dataset)} molecules")
        return nest_dataset

    def load_mols_dataset_dtwg(self, data_path, atoms, coords, dataset_type=1, **kwargs):
        """Load mols for offline encoding (optional LMDB V2 chunking)."""
        if dataset_type == 2:
            dataset = LMDBDatasetV2(data_path)
            keys = dataset.get_split("success")
            keys = list(sorted(list(set(keys))))
            start = kwargs.get("start", 0)
            end = kwargs.get("end")
            if end is None:
                end = len(keys)
            if start >= len(keys):
                raise ValueError(
                    "start should be less than len(keys) = {}".format(len(keys))
                )
            logger.info("chunk dataset, start: {}, end: {}".format(start, end))
            dataset.set_split("chunk", keys[start:end], deduplicate=False, temporary=True)
            dataset.set_default_split("chunk")
        else:
            if kwargs.get("start", 0) != 0 or kwargs.get("end", None) is not None:
                logger.info(
                    "chuck is not supported when using default lmdb, ignore start and end"
                )
            dataset = LMDBDataset(data_path)

        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )
        return self._build_mol_nest(dataset)

    def load_pockets_dataset(self, data_path, **kwargs):
        dataset = LMDBDataset(data_path)
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")
        dataset = self._prepare_pocket_apo(dataset)
        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")
        pocket_net, pocket_len = self._pocket_graph_from_apo(apo_dataset)
        return NestedDictionaryDataset(
            {
                "net_input": pocket_net,
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": pocket_len,
            },
        )

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)

        if args.finetune_mol_model is not None:
            logger.info("load pretrain model weight from %s", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)

        if args.finetune_pocket_model is not None:
            logger.info("load pretrain model weight from %s", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        return model

    def encode_mols_multi_folds(
        self,
        model,
        batch_size,
        mol_path,
        save_dir,
        use_cuda,
        dataset_type=None,
        write_npy=True,
        write_h5=True,
        flush_interval=60,
        fold_version="6_folds",
        **kwargs,
    ):
        ckpts = self._fold_checkpoints(fold_version)

        if dataset_type is None:
            dataset_type = 2 if os.path.isdir(mol_path) else 1
        logger.info(f"dataset_type: {dataset_type}")
        suffix = f"{kwargs.get('start', '')}{kwargs.get('end', '')}"
        if write_h5:
            h5_path = os.path.join(save_dir, f"mol_reps{suffix}.h5")
            logger.info(f"encoding write to {h5_path}, resume is supported")
        if write_npy:
            npy_path = os.path.join(save_dir, f"mol_reps{suffix}.npy")
            logger.info(
                f"encoding write to {npy_path} in one shot, "
                "embeddings will accumulate in the memory"
            )
            if not write_h5:
                logger.info("resume is not supported for npy")

        mol_reps_all = []
        n_folds = len(ckpts)
        # Historical H5 layout stores 128-d embeddings per fold concatenated.
        emb_dim = 128

        for fold, ckpt in enumerate(ckpts):
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)

            mol_dataset = self.load_mols_dataset_dtwg(
                mol_path, "atoms", "coordinates", dataset_type=dataset_type, **kwargs
            )
            collate_fn = mol_dataset.collater
            bsz = batch_size
            mol_reps = []
            hdf5 = None
            num_written = 0
            try:
                if write_h5:
                    hdf5 = h5py.File(
                        os.path.join(save_dir, f"mol_reps{suffix}.h5"), "a"
                    )
                    dset = hdf5.require_dataset(
                        "mol_reps",
                        shape=(len(mol_dataset), n_folds * emb_dim),
                        dtype=np.float32,
                        chunks=True,
                    )
                    kset = hdf5.require_dataset(
                        "fold{}".format(fold),
                        shape=(len(mol_dataset),),
                        dtype=np.bool_,
                        chunks=True,
                        compression="lzf",
                    )
                    written_mask = kset[:]
                    num_written = int(np.sum(written_mask))
                    if num_written == len(mol_dataset):
                        if write_npy:
                            mol_reps = dset[
                                :, fold * emb_dim : (fold + 1) * emb_dim
                            ]
                            mol_reps = np.expand_dims(mol_reps, axis=1)
                            mol_reps_all.append(mol_reps)
                        logger.info("Already written fold %s mols", fold)
                        continue
                    elif num_written > 0:
                        mol_dataset = torch.utils.data.Subset(
                            mol_dataset, range(num_written, len(mol_dataset))
                        )
                        if write_npy:
                            mol_reps.append(
                                dset[:num_written, fold * emb_dim : (fold + 1) * emb_dim]
                            )
                        logger.info(
                            "Already written %s mols in fold %s, will skip them",
                            num_written,
                            fold,
                        )
                logger.info(f"dataloader workers: {self.args.num_workers}")
                mol_data = torch.utils.data.DataLoader(
                    mol_dataset,
                    batch_size=bsz,
                    collate_fn=collate_fn,
                    num_workers=self.args.num_workers,
                )
                with torch.inference_mode():
                    for batch, sample in enumerate(tqdm(mol_data)):
                        if use_cuda:
                            sample = unicore.utils.move_to_cuda(sample)
                        mol_emb = self._encode_mol_batch(
                            model.mol_model, model.mol_project, sample
                        )
                        if write_h5:
                            sl = slice(
                                num_written + batch * bsz,
                                num_written + batch * bsz + len(mol_emb),
                            )
                            dset[sl, fold * emb_dim : (fold + 1) * emb_dim] = mol_emb
                            kset[sl] = 1
                            if batch % flush_interval == 0:
                                hdf5.flush()
                        if write_npy:
                            mol_reps.append(mol_emb)
                if write_npy:
                    mol_reps = np.concatenate(mol_reps, axis=0)
                    mol_reps = np.expand_dims(mol_reps, axis=1)
                    mol_reps_all.append(mol_reps)
            except Exception:
                logger.exception("encode_mols_multi_folds failed on fold %s", fold)
                if hdf5 is not None:
                    hdf5.close()
                raise
            finally:
                if hdf5 is not None:
                    hdf5.flush()
                    hdf5.close()

        if write_npy:
            mol_reps_all = np.concatenate(mol_reps_all, axis=1)
            mol_reps_all = mol_reps_all.astype(np.float32)
            logger.info("mol_reps shape %s", mol_reps_all.shape)
            np.save(
                os.path.join(save_dir, f"mol_reps{suffix}.npy"),
                mol_reps_all,
            )

    def encode_pockets_multi_folds(
        self,
        model,
        pocket_dir,
        pocket_path=None,
        fold_version="6_folds",
        use_cuda=True,
        **kwargs,
    ):
        # Backward compatible: callers pass (model, pocket_dir, pocket_path).
        # Only pocket_path is used for loading.
        if pocket_path is None:
            pocket_path = pocket_dir
        ckpts = self._fold_checkpoints(fold_version)
        logger.info("encoding pockets from %s", pocket_path)

        pocket_reps_all = []
        pocket_names = None
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = self._pocket_dataloader(pocket_dataset, use_cuda, batch_size=32)

        for fold, ckpt in enumerate(ckpts):
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)
            if pocket_names is None:
                pocket_reps, pocket_names = self._encode_pockets(
                    model, pocket_data, use_cuda, collect_names=True
                )
            else:
                pocket_reps = self._encode_pockets(model, pocket_data, use_cuda)
            pocket_reps_all.append(np.expand_dims(pocket_reps, axis=1))

        pocket_reps_all = np.concatenate(pocket_reps_all, axis=1)
        logger.info("pocket_reps shape %s", pocket_reps_all.shape)
        return pocket_reps_all, pocket_names

    def _encode_mol_batch_tensor(self, mol_model, mol_project, sample):
        # Run a single preprocessed batch through one fold's mol encoder and
        # return the (detached) embedding tensor still on its current device.
        # Used by the on-the-fly scoring path so the score matmul can run on the
        # GPU without ever copying the embeddings back to host.
        dist = sample["net_input"]["mol_src_distance"]
        et = sample["net_input"]["mol_src_edge_type"]
        st = sample["net_input"]["mol_src_tokens"]
        mol_padding_mask = st.eq(mol_model.padding_idx)
        mol_x = mol_model.embed_tokens(st)
        n_node = dist.size(-1)
        gbf_feature = mol_model.gbf(dist, et)
        gbf_result = mol_model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        mol_encoder_rep = mol_model.encoder.forward_repr(
            mol_x, padding_mask=mol_padding_mask, attn_mask=graph_attn_bias
        )[:, 0, :]
        mol_emb = mol_project(mol_encoder_rep)
        mol_emb = mol_emb / mol_emb.norm(dim=-1, keepdim=True)
        return mol_emb.detach()

    def _encode_mol_batch(self, mol_model, mol_project, sample):
        # Run a single preprocessed batch through one fold's mol encoder.
        # Keep the native dtype (fp16 on a half-precision GPU model). Halves the
        # bytes copied D2H and written to the cache; upcast happens transiently
        # at matmul time only.
        return self._encode_mol_batch_tensor(mol_model, mol_project, sample).cpu().numpy()

    def _input_file_cache_tag(self, data_path):
        # Stable identifier for an LMDB input file, derived from its absolute path
        # plus file size and mtime, so distinct inputs (or a regenerated file at
        # the same path) never reuse each other's cached embeddings.
        abspath = os.path.abspath(data_path)
        try:
            stat = os.stat(abspath)
            signature = f"{abspath}:{stat.st_size}:{int(stat.st_mtime)}"
        except OSError:
            signature = abspath
        digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:16]
        base = os.path.basename(os.path.normpath(data_path)) or "data"
        return f"{base}_{digest}"

    def _mol_library_cache_tag(self, mol_data_path):
        return f"{self._input_file_cache_tag(mol_data_path)}_{_MOL_LIB_CACHE_VERSION}"

    def _pocket_cache_dir(self, fold_version, pocket_path):
        pocket_tag = self._input_file_cache_tag(pocket_path)
        return os.path.join(
            f"./data/encoded_pocket_embs/{fold_version}",
            "pocket_cache",
            pocket_tag,
        )

    def _snapshot_mol_encoder(self, model, use_cuda, use_fp16):
        # Take a standalone, resident copy of this fold's mol encoder so it can be
        # reused after the next checkpoint overwrites the live model weights.
        # SDPA/FlashAttention is intentionally not used: forward_repr chains each
        # layer's pre-softmax scores as the next layer's attn_bias, which
        # FlashAttention cannot reproduce.
        fold_mol_model = copy.deepcopy(model.mol_model).eval()
        fold_mol_project = copy.deepcopy(model.mol_project).eval()
        if use_cuda:
            fold_mol_model = fold_mol_model.cuda()
            fold_mol_project = fold_mol_project.cuda()
        if use_fp16:
            fold_mol_model = fold_mol_model.half()
            fold_mol_project = fold_mol_project.half()
        return fold_mol_model, fold_mol_project

    def _load_fold_weights_into_encoder(self, resident, model):
        # Swap fold weights into a resident mol encoder without another deepcopy.
        # `model` must already hold the target fold's state_dict (via
        # load_checkpoint_to_cpu + load_state_dict).
        fold_mol_model, fold_mol_project = resident
        fold_mol_model.load_state_dict(model.mol_model.state_dict())
        fold_mol_project.load_state_dict(model.mol_project.state_dict())
        fold_mol_model.eval()
        fold_mol_project.eval()
        return resident

    def _close_mol_lmdb_envs(self, mol_dataset):
        # Walk nested/Subset wrappers and close any MolLMDBDataset envs so the
        # same path can be reopened (e.g. for opt-in late cascade compaction).
        seen = set()
        stack = [mol_dataset]
        while stack:
            ds = stack.pop()
            if ds is None or id(ds) in seen:
                continue
            seen.add(id(ds))
            if isinstance(ds, MolLMDBDataset):
                ds.close()
                continue
            for attr in ("dataset", "datasets"):
                child = getattr(ds, attr, None)
                if child is None:
                    continue
                if isinstance(child, dict):
                    stack.extend(child.values())
                elif isinstance(child, (list, tuple)):
                    stack.extend(child)
                else:
                    stack.append(child)

    def _encode_pockets(self, model, pocket_data, use_cuda, collect_names=False):
        # Encode all pockets for the currently loaded model weights.
        pocket_reps = []
        pocket_names = []
        with torch.inference_mode():
            for _, sample in enumerate(tqdm(pocket_data)):
                if use_cuda:
                    sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                pocket_padding_mask = st.eq(model.pocket_model.padding_idx)
                pocket_x = model.pocket_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket_model.gbf(dist, et)
                gbf_result = model.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_encoder_rep = model.pocket_model.encoder.forward_repr(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )[:, 0, :]
                pocket_emb = model.pocket_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().to(torch.float32).cpu().numpy()
                pocket_reps.append(pocket_emb)
                if collect_names:
                    pocket_names.extend(sample["pocket_name"])
        reps = np.concatenate(pocket_reps, axis=0).astype(np.float32)
        if collect_names:
            return reps, pocket_names
        return reps

    def _cache_pocket_reps_by_fold(
        self, model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, write_cache=True
    ):
        # Load per-fold pocket embeddings from disk when available; otherwise
        # encode once per fold and optionally persist them. Cache paths are keyed
        # by fold_version and the pocket LMDB path/size/mtime.
        n_folds = len(ckpts)
        cache_dir = self._pocket_cache_dir(fold_version, pocket_path)
        cache_paths = [os.path.join(cache_dir, f"fold{i}.pkl") for i in range(n_folds)]

        pocket_reps_by_fold = {}
        folds_to_encode = []
        for fold, cache_path in enumerate(cache_paths):
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    pocket_reps_by_fold[fold] = pickle.load(f)
            else:
                folds_to_encode.append(fold)

        if not folds_to_encode:
            logger.info(
                f"loaded pocket embeddings from {cache_dir} ({n_folds} fold(s))"
            )
            return pocket_reps_by_fold

        if pocket_reps_by_fold:
            logger.info(
                f"loaded pocket embeddings from {cache_dir} "
                f"({n_folds - len(folds_to_encode)}/{n_folds} fold(s)); "
                f"encoding missing fold(s): {folds_to_encode}"
            )
        else:
            logger.info(
                f"no pocket cache at {cache_dir}; encoding {n_folds} fold(s)"
            )

        for fold in folds_to_encode:
            ckpt = ckpts[fold]
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
            model.load_state_dict(state["model"], strict=False)
            pocket_reps_by_fold[fold] = self._encode_pockets(model, pocket_data, use_cuda)

        if write_cache:
            os.makedirs(cache_dir, exist_ok=True)
            for fold in folds_to_encode:
                with open(cache_paths[fold], "wb") as f:
                    pickle.dump(pocket_reps_by_fold[fold], f)
            logger.info(
                f"wrote pocket embeddings to {cache_dir} "
                f"({len(folds_to_encode)} fold(s))"
            )
        else:
            logger.info(
                f"cached pocket embeddings in memory for {n_folds} fold(s) "
                f"(write_cache=False, not persisted to {cache_dir})"
            )
        return pocket_reps_by_fold

    def _mol_dataloader_kwargs(self, use_cuda):
        # Shared DataLoader kwargs for molecule encoding/scoring paths.
        # persistent_workers=False: each scoring pass is a single epoch, and
        # leaving workers alive after iteration contends with CPU-heavy
        # rank_select (median/MAD / z-score) — worse with high --num-workers.
        num_workers = self.args.num_workers
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": use_cuda,
            "persistent_workers": False,
        }
        if num_workers > 0:
            prefetch = getattr(self.args, "prefetch_factor", 4)
            kwargs["prefetch_factor"] = max(2, int(prefetch) if prefetch else 4)
        return kwargs

    def _pocket_dataloader(self, pocket_dataset, use_cuda, batch_size=16):
        return torch.utils.data.DataLoader(
            pocket_dataset,
            batch_size=batch_size,
            collate_fn=pocket_dataset.collater,
            **self._mol_dataloader_kwargs(use_cuda),
        )

    @staticmethod
    def _shutdown_dataloader(loader):
        # Drop worker processes promptly so post-score CPU work is not starved.
        if loader is None:
            return
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()
        except Exception:
            pass
        gc.collect()

    def _resolve_collater(self, mol_dataset):
        # Walk Subset wrappers until a collater is found on NestedDictionaryDataset.
        collate_fn = None
        _ds = mol_dataset
        while _ds is not None and collate_fn is None:
            collate_fn = getattr(_ds, "collater", None)
            _ds = getattr(_ds, "dataset", None)
        if collate_fn is None:
            raise AttributeError("could not resolve collater for mol_dataset")
        return collate_fn

    def _score_mol_dataset(
        self,
        fold_encoders,
        pocket_reps_by_fold,
        mol_dataset,
        use_cuda,
        bsz,
        run_label=None,
        score_memmap=None,
        memmap_flush_interval=50,
        anchor_reservoir=None,
        gate_metric=None,
        gate_medians=None,
        gate_mads=None,
    ):
        # Stream a molecule dataset (the full library or a Subset) through one or
        # more resident fold encoders, computing the pocket-vs-molecule score
        # matrix on the fly. Only the (n_pockets x n_mols) score blocks are kept;
        # the per-molecule embeddings are scored and discarded batch by batch, so
        # the large (n_mols x emb_dim) arrays are never materialized.
        #
        #   fold_encoders:        {fold: (mol_model, mol_project)}
        #   pocket_reps_by_fold:  {fold: np.ndarray (n_pockets, emb_dim)}
        #   score_memmap:         optional np.memmap (n_pockets, n_mols) for a
        #                         *single* fold. When set, that fold's scores are
        #                         written to disk instead of a dense RAM array
        #                         (cascade large-library path). Other folds still
        #                         use dense storage.
        #   anchor_reservoir:     optional `_AnchorReservoir` updated from the
        #                         memmap fold's fp32 score batches (cascade).
        #   gate_metric:          optional (n_mols,) float32 filled from the single
        #                         scored fold's fp32 batches (cascade tier-0).
        #   gate_medians/mads:    optional anchors for gate_metric z-score; when
        #                         omitted, gate_metric uses max-over-pockets.
        # Returns (scores_by_fold, names) where scores_by_fold[fold] is either a
        # dense float32 array or the memmap handle when score_memmap was used.
        collate_fn = self._resolve_collater(mol_dataset)
        mol_data_loader = torch.utils.data.DataLoader(
            mol_dataset,
            batch_size=bsz,
            collate_fn=collate_fn,
            **self._mol_dataloader_kwargs(use_cuda),
        )
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        # Resident fp32 pocket tensors so the score matmul is done in full
        # precision regardless of the fp16 mol encoder, avoiding noise near the
        # top-k boundary.
        pocket_t_by_fold = {
            fold: torch.from_numpy(np.ascontiguousarray(reps)).to(device).float()
            for fold, reps in pocket_reps_by_fold.items()
        }
        n_mols = len(mol_dataset)
        n_folds = len(fold_encoders)
        n_iters = (n_mols + bsz - 1) // bsz
        label_suffix = f" ({run_label})" if run_label else ""
        if gate_metric is not None:
            if n_folds != 1:
                raise ValueError(
                    "gate_metric requires exactly one fold encoder, "
                    f"got {n_folds}"
                )
            if gate_metric.shape != (n_mols,):
                raise ValueError(
                    f"gate_metric shape {gate_metric.shape} does not match "
                    f"({n_mols},)"
                )
        memmap_fold = None
        if score_memmap is not None:
            if n_folds != 1:
                raise ValueError(
                    "score_memmap requires exactly one fold encoder, "
                    f"got {n_folds}"
                )
            memmap_fold = next(iter(fold_encoders))
            if score_memmap.shape != (
                pocket_reps_by_fold[memmap_fold].shape[0],
                n_mols,
            ):
                raise ValueError(
                    f"score_memmap shape {score_memmap.shape} does not match "
                    f"({pocket_reps_by_fold[memmap_fold].shape[0]}, {n_mols})"
                )
            sink_bits = [f"score_sink=memmap({score_memmap.dtype})"]
            if gate_metric is not None:
                sink_bits.append("gate_metric=float32")
            logger.info(
                f"mol encoding{label_suffix}: iterations={n_iters}, folds={n_folds}, "
                f"batch_size={bsz}, molecules={n_mols}, "
                f"{', '.join(sink_bits)}"
            )
        else:
            extra = ", gate_metric=float32" if gate_metric is not None else ""
            logger.info(
                f"mol encoding{label_suffix}: iterations={n_iters}, folds={n_folds}, "
                f"batch_size={bsz}, molecules={n_mols}{extra}"
            )
        scores_by_fold = {}
        names_acc = []
        offset = 0
        batches_since_flush = 0
        metric_fold = (
            next(iter(fold_encoders)) if gate_metric is not None else None
        )
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                for _, sample in enumerate(tqdm(mol_data_loader)):
                    if use_cuda:
                        sample = unicore.utils.move_to_cuda(sample)
                    batch_len = 0
                    for fold, (fold_mol_model, fold_mol_project) in fold_encoders.items():
                        emb = self._encode_mol_batch_tensor(
                            fold_mol_model, fold_mol_project, sample
                        )
                        emb = emb.float()
                        pocket_t = pocket_t_by_fold[fold]
                        score = pocket_t @ emb.t()
                        score = score.cpu().numpy().astype(np.float32)
                        batch_len = score.shape[1]
                        if fold == memmap_fold:
                            if anchor_reservoir is not None:
                                anchor_reservoir.add_batch(score, offset)
                            if score_memmap.dtype == np.float16:
                                score_memmap[:, offset:offset + batch_len] = (
                                    score.astype(np.float16, copy=False)
                                )
                            else:
                                score_memmap[:, offset:offset + batch_len] = score
                        else:
                            if fold not in scores_by_fold:
                                scores_by_fold[fold] = np.empty(
                                    (score.shape[0], n_mols), dtype=np.float32
                                )
                            scores_by_fold[fold][
                                :, offset:offset + batch_len
                            ] = score
                            # Dense single-fold cascade: feed reservoir from the
                            # only scored fold (memmap_fold is None).
                            if (
                                anchor_reservoir is not None
                                and memmap_fold is None
                                and n_folds == 1
                            ):
                                anchor_reservoir.add_batch(score, offset)
                        if fold == metric_fold:
                            gate_metric[offset:offset + batch_len] = (
                                _batch_gate_metric(
                                    score, gate_medians, gate_mads
                                )
                            )
                    names_acc.extend(sample["smi_name"])
                    offset += batch_len
                    if memmap_fold is not None:
                        batches_since_flush += 1
                        if batches_since_flush >= memmap_flush_interval:
                            score_memmap.flush()
                            batches_since_flush = 0
            if use_cuda:
                torch.cuda.synchronize()
        finally:
            # Reap DataLoader workers before caller runs rank_select / median-MAD.
            self._shutdown_dataloader(mol_data_loader)
        if memmap_fold is not None:
            score_memmap.flush()
            scores_by_fold[memmap_fold] = score_memmap
        elapsed = time.perf_counter() - t0
        mols_per_sec = n_mols / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"mol encoding{label_suffix}: {mols_per_sec:.1f} molecules/s "
            f"({elapsed:.2f}s for {n_mols} molecules)"
        )
        return scores_by_fold, names_acc

    def _score_memmap_cache_dir(self, fold_version, mol_data_path, pocket_path):
        # Scratch/cache dir for the fold-mean score memmap used by streaming full
        # mode. Keyed by fold_version + mol library tag + pocket tag so distinct
        # inputs never collide.
        mol_tag = self._mol_library_cache_tag(mol_data_path)
        pocket_tag = self._input_file_cache_tag(pocket_path)
        return os.path.join(
            f"./data/encoded_mol_embs/{fold_version}",
            "score_memmap",
            f"{mol_tag}__{pocket_tag}",
        )

    def _score_memmap_paths(self, cache_dir):
        return {
            "dir": cache_dir,
            "scores": os.path.join(cache_dir, "foldmean_scores.dat"),
            "meta": os.path.join(cache_dir, "meta.npz"),
            "names": os.path.join(cache_dir, "names.npy"),
            "names_txt": os.path.join(cache_dir, "names.txt"),
        }

    def _open_or_create_score_memmap(self, cache_dir, n_pockets, n_mols, dtype=None):
        # Open an existing fold-mean score memmap or create a fresh one. Returns
        # (memmap, n_written, paths). Resume is supported via meta.npz.
        if dtype is None:
            dtype = _score_memmap_dtype()
        paths = self._score_memmap_paths(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        n_written = 0
        if os.path.exists(paths["meta"]) and os.path.exists(paths["scores"]):
            try:
                meta = np.load(paths["meta"])
                meta_pockets = int(meta["n_pockets"])
                meta_mols = int(meta["n_mols"])
                meta_written = int(meta["n_written"])
                meta_dtype = np.dtype(str(meta["dtype"]))
                if (
                    meta_pockets == n_pockets
                    and meta_mols == n_mols
                    and meta_dtype == np.dtype(dtype)
                ):
                    n_written = max(0, min(meta_written, n_mols))
                    mode = "r+" if n_written < n_mols else "r"
                    memmap = np.memmap(
                        paths["scores"], dtype=dtype, mode=mode, shape=(n_pockets, n_mols)
                    )
                    logger.info(
                        f"opened score memmap at {cache_dir} "
                        f"(written={n_written}/{n_mols}, dtype={dtype})"
                    )
                    return memmap, n_written, paths
                logger.warning(
                    f"score memmap meta mismatch at {cache_dir}; recreating"
                )
            except Exception as e:
                logger.warning(f"failed to open score memmap at {cache_dir}: {e}")

        if os.path.exists(paths["scores"]):
            try:
                os.remove(paths["scores"])
            except OSError:
                pass
        memmap = np.memmap(
            paths["scores"], dtype=dtype, mode="w+", shape=(n_pockets, n_mols)
        )
        np.savez(
            paths["meta"],
            n_pockets=n_pockets,
            n_mols=n_mols,
            n_written=0,
            dtype=np.array(str(np.dtype(dtype))),
        )
        # Truncate any stale names sidecar from a previous incomplete run.
        for key in ("names", "names_txt"):
            if os.path.exists(paths[key]):
                try:
                    os.remove(paths[key])
                except OSError:
                    pass
        logger.info(
            f"created score memmap at {cache_dir} "
            f"(shape=({n_pockets}, {n_mols}), dtype={dtype})"
        )
        return memmap, 0, paths

    def _flush_score_memmap_meta(self, paths, n_pockets, n_mols, n_written, dtype):
        np.savez(
            paths["meta"],
            n_pockets=n_pockets,
            n_mols=n_mols,
            n_written=n_written,
            dtype=np.array(str(np.dtype(dtype))),
        )

    def _load_names_sidecar(self, paths, n_mols):
        if os.path.exists(paths["names"]):
            names = np.load(paths["names"], allow_pickle=True)
            if len(names) == n_mols:
                return list(names.tolist())
        if os.path.exists(paths["names_txt"]):
            with open(paths["names_txt"], "r") as f:
                names = [line.rstrip("\n") for line in f]
            if len(names) == n_mols:
                return names
        return None

    def _write_names_sidecar(self, paths, names):
        np.save(paths["names"], np.asarray(names, dtype=object))
        with open(paths["names_txt"], "w") as f:
            for name in names:
                f.write(f"{name}\n")

    def _fused_foldmean_score_pass(
        self,
        fold_encoders,
        pocket_reps_by_fold,
        mol_dataset,
        use_cuda,
        bsz,
        memmap,
        paths,
        n_written=0,
        run_label=None,
        flush_interval=50,
    ):
        # Stream the molecule library through all resident fold encoders in one
        # DataLoader pass. Per batch, compute the fold-mean pocket@mol score on
        # GPU and write columns into the on-disk memmap. Never materializes
        # (n_mols x emb_dim) embeddings or a full (n_pockets x n_mols) host array.
        n_mols = len(mol_dataset)
        n_pockets = memmap.shape[0]
        dtype = memmap.dtype
        if n_written >= n_mols:
            names = self._load_names_sidecar(paths, n_mols)
            if names is None:
                raise RuntimeError(
                    f"score memmap complete ({n_mols}) but names sidecar missing "
                    f"at {paths['dir']}"
                )
            return names

        collate_fn = self._resolve_collater(mol_dataset)
        if n_written > 0:
            encode_dataset = torch.utils.data.Subset(
                mol_dataset, range(n_written, n_mols)
            )
            names_acc = self._load_names_sidecar(paths, n_written)
            if names_acc is None:
                # Rebuild the already-written names by reading the dataset head
                # (no GPU work). Rare; only when meta exists without names.
                logger.info(
                    f"rebuilding names for first {n_written} molecules from dataset"
                )
                head = torch.utils.data.Subset(mol_dataset, range(n_written))
                head_loader = torch.utils.data.DataLoader(
                    head,
                    batch_size=bsz,
                    collate_fn=collate_fn,
                    **self._mol_dataloader_kwargs(use_cuda),
                )
                names_acc = []
                for sample in head_loader:
                    names_acc.extend(sample["smi_name"])
            else:
                names_acc = list(names_acc)
        else:
            encode_dataset = mol_dataset
            names_acc = []

        mol_data_loader = torch.utils.data.DataLoader(
            encode_dataset,
            batch_size=bsz,
            collate_fn=collate_fn,
            **self._mol_dataloader_kwargs(use_cuda),
        )
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        pocket_t_by_fold = {
            fold: torch.from_numpy(np.ascontiguousarray(reps)).to(device).float()
            for fold, reps in pocket_reps_by_fold.items()
        }
        n_folds = len(fold_encoders)
        remaining = n_mols - n_written
        n_iters = (remaining + bsz - 1) // bsz
        label_suffix = f" ({run_label})" if run_label else ""
        logger.info(
            f"fused fold-mean scoring{label_suffix}: iterations={n_iters}, "
            f"folds={n_folds}, batch_size={bsz}, "
            f"molecules={remaining} (resume_from={n_written})"
        )
        offset = n_written
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for batch_idx, sample in enumerate(tqdm(mol_data_loader)):
                if use_cuda:
                    sample = unicore.utils.move_to_cuda(sample)
                mean_score = None
                for fold, (fold_mol_model, fold_mol_project) in fold_encoders.items():
                    emb = self._encode_mol_batch_tensor(
                        fold_mol_model, fold_mol_project, sample
                    ).float()
                    score = pocket_t_by_fold[fold] @ emb.t()
                    if mean_score is None:
                        mean_score = score
                    else:
                        mean_score = mean_score + score
                mean_score = mean_score / float(n_folds)
                score_np = mean_score.detach().cpu().numpy().astype(dtype, copy=False)
                batch_len = score_np.shape[1]
                memmap[:, offset:offset + batch_len] = score_np
                names_acc.extend(sample["smi_name"])
                offset += batch_len
                if batch_idx % flush_interval == 0:
                    memmap.flush()
                    self._flush_score_memmap_meta(
                        paths, n_pockets, n_mols, offset, dtype
                    )
        if use_cuda:
            torch.cuda.synchronize()
        memmap.flush()
        self._flush_score_memmap_meta(paths, n_pockets, n_mols, offset, dtype)
        self._write_names_sidecar(paths, names_acc)
        elapsed = time.perf_counter() - t0
        mols_per_sec = remaining / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"fused fold-mean scoring{label_suffix}: {mols_per_sec:.1f} molecules/s "
            f"({elapsed:.2f}s for {remaining} molecules)"
        )
        if offset != n_mols:
            raise RuntimeError(
                f"fused score pass wrote {offset} columns, expected {n_mols}"
            )
        return names_acc

    def _fill_score_memmap_from_pickles(
        self,
        mol_reps_by_fold,
        pocket_reps_by_fold,
        names,
        memmap,
        paths,
        use_cuda,
        chunk_size=None,
    ):
        # Chunked fold-mean matmul from pre-encoded per-fold embeddings into the
        # score memmap. Avoids uploading the whole library (or allocating a full
        # score matrix) at once. Used by the use_cache=True path.
        if chunk_size is None:
            chunk_size = _SCORE_MEMMAP_SELECT_CHUNK
        n_folds = len(mol_reps_by_fold)
        n_pockets, n_mols = memmap.shape
        dtype = memmap.dtype
        device = torch.device("cuda") if use_cuda else torch.device("cpu")
        pocket_t_by_fold = {
            fold: torch.from_numpy(
                np.ascontiguousarray(pocket_reps_by_fold[fold])
            ).to(device).float()
            for fold in mol_reps_by_fold
        }
        logger.info(
            f"filling score memmap from pickles: folds={n_folds}, "
            f"molecules={n_mols}, chunk={chunk_size}, cuda={use_cuda}"
        )
        for start in tqdm(range(0, n_mols, chunk_size), desc="pickle->memmap"):
            end = min(n_mols, start + chunk_size)
            mean_score = None
            for fold, reps in mol_reps_by_fold.items():
                mol_chunk = np.ascontiguousarray(reps[start:end]).astype(np.float32, copy=False)
                if use_cuda:
                    mol_t = torch.from_numpy(mol_chunk).to(device)
                    score = pocket_t_by_fold[fold] @ mol_t.t()
                    score = score.detach().cpu().numpy()
                    del mol_t
                else:
                    score = pocket_reps_by_fold[fold].astype(np.float32) @ mol_chunk.T
                if mean_score is None:
                    mean_score = score
                else:
                    mean_score = mean_score + score
            mean_score = (mean_score / float(n_folds)).astype(dtype, copy=False)
            memmap[:, start:end] = mean_score
        memmap.flush()
        self._flush_score_memmap_meta(paths, n_pockets, n_mols, n_mols, dtype)
        self._write_names_sidecar(paths, names)
        logger.info(f"wrote fold-mean score memmap ({n_pockets}x{n_mols}) from pickles")

    def _select_topk_from_score_memmap(
        self, memmap, fold_version, names, save_path, chunk_size=None
    ):
        # Exact per-pocket median/MAD from contiguous memmap rows, then
        # molecule-chunked z-score + max-over-pockets into a (n_mols,) vector,
        # then argpartition top-k. Matches `_ensemble_rank_mols` numerically
        # (aside from on-disk score dtype quantization).
        if chunk_size is None:
            chunk_size = _SCORE_MEMMAP_SELECT_CHUNK
        n_pockets, n_mols = memmap.shape
        do_zscore = fold_version.startswith("6_folds")
        if do_zscore:
            logger.info(
                f"computing per-pocket median/MAD over {n_mols} molecules "
                f"({n_pockets} pockets)"
            )
            medians, mads = _robust_pocket_anchors_memmap(memmap)
            logger.info(
                f"selection pass over score memmap: chunk={chunk_size}, "
                f"zscore={do_zscore}"
            )
            res_max = _max_zscore_metric(
                memmap, medians, mads, chunk_size=chunk_size
            )
        else:
            res_max = np.empty(n_mols, dtype=np.float32)
            logger.info(
                f"selection pass over score memmap: chunk={chunk_size}, "
                f"zscore={do_zscore}"
            )
            for start in tqdm(range(0, n_mols, chunk_size), desc="select"):
                end = min(n_mols, start + chunk_size)
                block = np.asarray(memmap[:, start:end], dtype=np.float32)
                res_max[start:end] = np.max(block, axis=0)

        k = max(1, int(n_mols * RETRIEVAL_TOP_FRAC))
        if k > 0:
            top_idx = np.argpartition(res_max, -k)[-k:]
            top_idx = top_idx[np.argsort(res_max[top_idx])[::-1]]
        else:
            top_idx = np.empty(0, dtype=np.int64)

        with open(save_path, "w") as f:
            for i in top_idx:
                f.write(f"{names[i]},{res_max[i]}\n")
        logger.info(
            f"wrote top {k}/{n_mols} ({RETRIEVAL_TOP_FRAC:.2%}) results to {save_path}"
        )
        return res_max, top_idx

    def benchmark_mol_encoding(
        self,
        model,
        mol_data_path,
        use_cuda,
        batch_sizes=(64, 128, 256, 512, 1024),
        num_workers_list=None,
        num_folds=6,
        max_mols=20000,
        warmup_batches=3,
        repeats=1,
        **kwargs,
    ):
        # Sweep DataLoader batch size / num_workers for the first-run molecule
        # encoding path (the uncached branch of retrieval_multi_folds) and
        # report end-to-end throughput.
        #
        # `num_folds` resident copies of the current mol encoder are snapshotted
        # to reproduce the memory pressure and per-batch compute of an uncached
        # multi-fold run: each library batch is pushed through every resident fold
        # encoder, exactly as the real pass does. Timing covers the full pipeline
        # (LMDB read + collate + H2D + forward + D2H), since that is the throughput
        # that actually governs wall-clock time.
        import time

        if num_workers_list is None:
            num_workers_list = [self.args.num_workers]

        use_fp16 = next(model.parameters()).dtype == torch.float16

        # Snapshot fold encoders once and reuse them across every configuration so
        # the measured differences come from the DataLoader settings rather than
        # from per-config setup cost.
        fold_encoders = {
            fold: self._snapshot_mol_encoder(model, use_cuda, use_fp16)
            for fold in range(num_folds)
        }

        full_dataset = self.load_mols_dataset(
            mol_data_path, "atoms", "coordinates", readahead=True
        )
        collate_fn = full_dataset.collater
        n_total = len(full_dataset)
        n_bench = min(max_mols, n_total) if max_mols else n_total
        if n_bench < n_total:
            bench_dataset = torch.utils.data.Subset(full_dataset, range(n_bench))
        else:
            bench_dataset = full_dataset

        logger.info(
            f"benchmarking mol encoding: {n_bench}/{n_total} mols, "
            f"{num_folds} resident fold encoder(s), fp16={use_fp16}, "
            f"warmup_batches={warmup_batches}, repeats={repeats}"
        )

        def run_one(bsz, num_workers):
            # Returns (mols_per_sec, peak_vram_gb, status). Times only the batches
            # after the warmup window using a single sync at each boundary to avoid
            # per-batch synchronization overhead skewing the result.
            if use_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            n_batches = (n_bench + bsz - 1) // bsz
            effective_warmup = warmup_batches if n_batches > warmup_batches + 1 else 0
            loader_kwargs = {
                "num_workers": num_workers,
                "pin_memory": use_cuda,
                "persistent_workers": False,
            }
            if num_workers > 0:
                prefetch = getattr(self.args, "prefetch_factor", 4)
                loader_kwargs["prefetch_factor"] = max(
                    2, int(prefetch) if prefetch else 4
                )
            loader = torch.utils.data.DataLoader(
                bench_dataset,
                batch_size=bsz,
                collate_fn=collate_fn,
                **loader_kwargs,
            )
            timed_mols = 0
            t0 = None
            with torch.no_grad():
                for batch_idx, sample in enumerate(loader):
                    if batch_idx == effective_warmup:
                        if use_cuda:
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                    if use_cuda:
                        sample = unicore.utils.move_to_cuda(sample)
                    for _, (fold_mol_model, fold_mol_project) in fold_encoders.items():
                        self._encode_mol_batch(fold_mol_model, fold_mol_project, sample)
                    if batch_idx >= effective_warmup:
                        timed_mols += sample["net_input"]["mol_src_tokens"].size(0)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0 if t0 is not None else float("nan")
            peak_vram = (
                torch.cuda.max_memory_allocated() / 1e9 if use_cuda else float("nan")
            )
            mps = timed_mols / elapsed if elapsed and elapsed > 0 else float("nan")
            return mps, peak_vram

        results = []
        for num_workers in num_workers_list:
            for bsz in batch_sizes:
                try:
                    best_mps, peak_vram = -1.0, float("nan")
                    for _ in range(max(1, repeats)):
                        mps, peak_vram = run_one(bsz, num_workers)
                        # Report the best (most favorable) run to reduce noise from
                        # transient OS/disk contention.
                        if mps == mps and mps > best_mps:
                            best_mps = mps
                    results.append((num_workers, bsz, best_mps, peak_vram, "ok"))
                    logger.info(
                        f"  workers={num_workers:>2} bsz={bsz:>5} "
                        f"-> {best_mps:8.1f} mols/s, peak {peak_vram:5.2f} GB"
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if use_cuda:
                            torch.cuda.empty_cache()
                        results.append(
                            (num_workers, bsz, float("nan"), float("nan"), "OOM")
                        )
                        logger.info(f"  workers={num_workers:>2} bsz={bsz:>5} -> OOM")
                    else:
                        raise

        gpu_name = torch.cuda.get_device_name() if use_cuda else "CPU"
        print("\n=== mol-encoding throughput sweep ===")
        print(f"device: {gpu_name}")
        print(f"mols benchmarked: {n_bench} | resident folds: {num_folds} | fp16: {use_fp16}")
        print(f"{'workers':>7} | {'bsz':>6} | {'mols/s':>10} | {'peak_VRAM_GB':>12} | status")
        print("-" * 58)
        best = None
        for num_workers, bsz, mps, peak_vram, status in results:
            mps_str = f"{mps:10.1f}" if mps == mps else f"{'-':>10}"
            vram_str = f"{peak_vram:12.2f}" if peak_vram == peak_vram else f"{'-':>12}"
            print(f"{num_workers:>7} | {bsz:>6} | {mps_str} | {vram_str} | {status}")
            if status == "ok" and (best is None or mps > best[2]):
                best = (num_workers, bsz, mps, peak_vram, status)
        if best is not None:
            print("-" * 58)
            print(
                f"best: workers={best[0]} bsz={best[1]} "
                f"-> {best[2]:.1f} mols/s (peak {best[3]:.2f} GB)"
            )
        return results

    def retrieval_multi_folds(self, model, pocket_path, save_path, mol_data_path, fold_version, use_cache=True, use_cuda=True, retrieval_mode="full", cascade_frac=0.2, cascade_tier_fracs=None, cascade_gate_folds=None, write_cache=True, retrieval_bsz=None, **kwargs):
        ckpts = self._fold_checkpoints(fold_version)
        caches = self._mol_cache_paths(fold_version)

        # Encode pockets once (cheap) up front; both retrieval modes need them.
        # load datasets once outside the fold loop to avoid re-opening the same lmdb environment
        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = self._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)

        if retrieval_mode == "cascade":
            # Multi-tier cascade screening: a configurable number of single-fold
            # gating tiers progressively narrows the library, then the surviving
            # pool is re-scored through every fold and ranked with the same
            # procedure as full mode. Never writes per-fold caches (gating folds
            # only see a subset, so the standard cache format would be misleading).
            self._retrieval_cascade(
                model, ckpts, pocket_data, save_path, mol_data_path,
                fold_version, cascade_frac, cascade_tier_fracs, cascade_gate_folds,
                pocket_path, use_cuda, retrieval_bsz, write_cache,
            )
            return

        # Streaming full mode: one encode (or chunked pickle matmul) pass writes
        # fold-mean scores to an on-disk memmap; a cheap CPU pass then computes
        # exact per-pocket median/MAD, max-over-pockets, and top-k. Never holds
        # (n_mols x emb_dim) or (n_pockets x n_mols) arrays in RAM/VRAM.
        use_fp16 = next(model.parameters()).dtype == torch.float16
        bsz = retrieval_bsz if retrieval_bsz and retrieval_bsz > 0 else _DEFAULT_RETRIEVAL_BSZ_FULL

        pocket_reps_by_fold = self._cache_pocket_reps_by_fold(
            model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, write_cache
        )
        n_pockets = next(iter(pocket_reps_by_fold.values())).shape[0]
        n_folds = len(ckpts)

        # Prefer a complete fold-mean score memmap cache when present.
        score_cache_dir = self._score_memmap_cache_dir(
            fold_version, mol_data_path, pocket_path
        )
        # Resolve n_mols cheaply: dataset length, or a single pickle's name list,
        # or an existing meta.npz — avoid loading all fold pickles up front.
        mol_dataset = None
        mol_reps_by_fold = {}
        mol_names = None
        if use_cache:
            paths_probe = self._score_memmap_paths(score_cache_dir)
            n_mols = None
            if os.path.exists(paths_probe["meta"]):
                try:
                    meta = np.load(paths_probe["meta"])
                    if int(meta["n_pockets"]) == n_pockets:
                        n_mols = int(meta["n_mols"])
                except Exception:
                    n_mols = None
            if n_mols is None:
                # Load fold-0 pickle just for library size / names.
                if not os.path.exists(caches[0]):
                    raise FileNotFoundError(
                        f"use_cache=True but missing mol cache: {caches[0]}"
                    )
                with open(caches[0], "rb") as f:
                    _, mol_names = pickle.load(f)
                n_mols = len(mol_names)
        else:
            mol_dataset = self.load_mols_dataset(
                mol_data_path, "atoms", "coordinates", readahead=True
            )
            n_mols = len(mol_dataset)

        memmap, n_written, paths = self._open_or_create_score_memmap(
            score_cache_dir, n_pockets, n_mols
        )

        filled_this_run = False
        if n_written >= n_mols:
            names = self._load_names_sidecar(paths, n_mols)
            if names is None:
                if mol_names is not None:
                    names = list(mol_names)
                    self._write_names_sidecar(paths, names)
                elif use_cache:
                    with open(caches[0], "rb") as f:
                        _, names = pickle.load(f)
                    names = list(names)
                    self._write_names_sidecar(paths, names)
                else:
                    raise RuntimeError(
                        f"complete score memmap at {score_cache_dir} but names missing"
                    )
            logger.info(
                f"reusing complete score memmap at {score_cache_dir} "
                f"({n_pockets}x{n_mols}); skipping encode"
            )
        elif use_cache:
            # Load all fold pickles only when we actually need to fill the memmap.
            for fold in range(n_folds):
                mol_cache_path = caches[fold]
                if not os.path.exists(mol_cache_path):
                    raise FileNotFoundError(
                        f"use_cache=True but missing mol cache: {mol_cache_path}"
                    )
                with open(mol_cache_path, "rb") as f:
                    reps, names = pickle.load(f)
                mol_reps_by_fold[fold] = reps
                mol_names = names
            self._fill_score_memmap_from_pickles(
                mol_reps_by_fold,
                pocket_reps_by_fold,
                mol_names,
                memmap,
                paths,
                use_cuda,
            )
            names = list(mol_names)
            mol_reps_by_fold.clear()
            filled_this_run = True
        else:
            # Snapshot every fold encoder once, then one fused DataLoader pass.
            fold_encoders = {}
            for fold, ckpt in enumerate(ckpts):
                state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
                model.load_state_dict(state["model"], strict=False)
                fold_encoders[fold] = self._snapshot_mol_encoder(
                    model, use_cuda, use_fp16
                )
            names = self._fused_foldmean_score_pass(
                fold_encoders,
                pocket_reps_by_fold,
                mol_dataset,
                use_cuda,
                bsz,
                memmap,
                paths,
                n_written=n_written,
                run_label="full retrieval",
            )
            fold_encoders.clear()
            if use_cuda:
                torch.cuda.empty_cache()
            filled_this_run = True

        self._select_topk_from_score_memmap(memmap, fold_version, names, save_path)

        # Release the memmap handle. Delete scratch files only when this run
        # filled them and the caller disabled persistence.
        del memmap
        if filled_this_run and not write_cache:
            for key in ("scores", "meta", "names", "names_txt"):
                try:
                    if os.path.exists(paths[key]):
                        os.remove(paths[key])
                except OSError:
                    pass
        return

    def _subset_mol_dataset(self, mol_dataset, indices):
        # Build a collate-capable Subset over global source indices (numpy ok).
        subset = torch.utils.data.Subset(mol_dataset, indices)
        subset.collater = self._resolve_collater(mol_dataset)
        return subset

    @staticmethod
    def _cascade_compact_enabled():
        # Opt-in late compaction before rescore (HDD seek storms). Default off so
        # gate tiers use Subset-on-source with no inter-tier LMDB copy.
        return os.environ.get("DRUGCLIP_CASCADE_COMPACT", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _retrieval_cascade(self, model, ckpts, pocket_data, save_path, mol_data_path, fold_version, cascade_frac, cascade_tier_fracs, cascade_gate_folds, pocket_path, use_cuda, retrieval_bsz, write_cache=True):
        # Two-phase cascade screening.
        #
        # Phase 1 (gating / first screening): a configurable number of
        # single-fold tiers progressively narrows the library. Tier t scores one
        # fold over the current pool and keeps the top
        # `cascade_frac * cascade_tier_fracs[t]` fraction (relative to the full
        # library), ranked by the running mean ensemble of the gating folds
        # (z-scored with the first gating fold's full-library anchor for
        # multi-pocket targets) and max-over-pockets.
        #
        # Gate tiers score a sorted Subset of the source LMDB (no inter-tier
        # compaction). Optional DRUGCLIP_CASCADE_COMPACT=1 materializes the final
        # gate pool into a sequential temp LMDB before Phase 2 only.
        #
        # Phase 2 (rescore / second screening): the surviving pool is ranked with
        # the same procedure as full mode (`_ensemble_rank_mols`) over all folds.
        # The gating folds reuse the scores already computed during Phase 1
        # (numerically identical, since each molecule is encoded independently),
        # so only the remaining folds are encoded here, in a single DataLoader
        # pass. Scores are computed on the fly, so per-fold (n_mols x emb_dim)
        # embedding matrices are never materialized.
        n_folds = len(ckpts)
        base_bsz = (
            retrieval_bsz
            if retrieval_bsz and retrieval_bsz > 0
            else _DEFAULT_RETRIEVAL_BSZ_CASCADE
        )
        use_fp16 = next(model.parameters()).dtype == torch.float16
        do_late_compact = self._cascade_compact_enabled()

        # Sequential full-library scan: enable LMDB readahead for HDD.
        mol_dataset = self.load_mols_dataset(
            mol_data_path, "atoms", "coordinates", readahead=True
        )

        # Each gating tier keeps cascade_frac * tier_fracs[t] of the full library.
        tier_fracs = list(cascade_tier_fracs) if cascade_tier_fracs else [1.0, 0.5, 0.25]
        if any(m <= 0 for m in tier_fracs):
            raise ValueError(f"cascade_tier_fracs must be positive, got {tier_fracs}")
        # At most one gating tier per available fold; the rescore uses every fold.
        n_gate_tiers = min(len(tier_fracs), n_folds)

        # Folds consumed by the gating tiers, in order. Start from the requested
        # gate folds, then pad with any remaining folds in ascending order so
        # there is always a fold available for every tier.
        requested_gate_folds = list(cascade_gate_folds) if cascade_gate_folds else [4, 1]
        if any(f < 0 or f >= n_folds for f in requested_gate_folds):
            raise ValueError(
                f"cascade_gate_folds out of range for {n_folds} folds: {requested_gate_folds}"
            )
        fold_order = [f for f in requested_gate_folds if f < n_folds]
        fold_order += [f for f in range(n_folds) if f not in fold_order]

        # Folds gated in Phase 1 are reused in the rescore; only the rest are
        # re-encoded there (see Phase 2 below).
        gate_folds = set(fold_order[:n_gate_tiers])
        rescore_folds = [f for f in range(n_folds) if f not in gate_folds]

        gate_bsz = _cascade_tier_bsz(base_bsz, n_folds, 1)
        # The rescore pass only runs the non-gating folds per batch, so scale the
        # batch size up to keep per-batch work roughly constant. This only affects
        # DataLoader batching, not the scores (mol embeddings are batch-invariant).
        rescore_bsz = (
            _cascade_tier_bsz(base_bsz, n_folds, len(rescore_folds))
            if rescore_folds
            else base_bsz
        )

        def pool_size(mult, prev, round_bsz):
            # Tier sizes are relative to the FULL library and nested in prev.
            # Round up to a full batch so the next tier's DataLoader sees complete batches.
            exact = max(1, int(n_mols * cascade_frac * mult))
            return min(prev, _ceil_to_batch(exact, round_bsz))

        def next_pool_round_bsz(completed_gate_tier):
            # Rounding batch size for the pool that the next pass will read.
            # The final surviving pool size must stay independent of the rescore
            # batch size: _ensemble_rank_mols z-scores each pocket over the pool,
            # so changing the pool size would shift every output score. Round the
            # final pool to base_bsz to match full mode / the reference pool.
            if completed_gate_tier + 1 < n_gate_tiers:
                return gate_bsz
            return base_bsz

        # Encode pockets once per fold up front; reuse for every scoring pass.
        pocket_reps_by_fold = self._cache_pocket_reps_by_fold(
            model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, write_cache
        )

        cascade_tmp_dir = None
        score_tmp_dir = None
        score_memmap_path = None
        score_memmap = None
        compact_dataset = None
        gate_encoder = None
        score_dataset = mol_dataset

        try:
            # Phase A: first tier fold over the FULL library.
            first_fold = fold_order[0]
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpts[first_fold])
            model.load_state_dict(state["model"], strict=False)
            # Snapshot once; later gate tiers swap weights into this resident copy.
            gate_encoder = self._snapshot_mol_encoder(model, use_cuda, use_fp16)

            n_mols = len(mol_dataset)
            n_pock = int(pocket_reps_by_fold[first_fold].shape[0])
            use_score_memmap = _cascade_scores_need_memmap(n_pock, n_mols)
            score_dtype = _cascade_score_memmap_dtype()
            # First-tier fold full-library robust stats, used as the z-score
            # normalization anchor at every tier. Skip when unused: for
            # single-pocket targets do_zscore is False.
            do_zscore = fold_version.startswith("6_folds") and n_pock > 1
            anchor_sample = _cascade_anchor_sample_size() if do_zscore else 0
            medians_anchor = None
            mads_anchor = None
            anchor_prepass_s = 0.0
            gate_metric = None

            # Prefer a dedicated score-memmap root so compact (CASCADE_TMP) can
            # live on a different device and avoid writeback contention.
            score_tmp_root = (
                os.environ.get("DRUGCLIP_SCORE_MEMMAP_DIR")
                or os.environ.get("DRUGCLIP_CASCADE_TMP")
                or tempfile.gettempdir()
            )
            compact_tmp_root = (
                os.environ.get("DRUGCLIP_CASCADE_TMP")
                or tempfile.gettempdir()
            )

            if do_zscore and anchor_sample > 0:
                # Reservoir pre-pass: score the fixed subsample first so anchors
                # exist before the full-library encode (enables metric-during-encode).
                anchor_reservoir = _AnchorReservoir(
                    n_pock, n_mols, sample_size=anchor_sample, seed=1
                )
                logger.info(
                    f"cascade tier 0 anchor reservoir: sample={anchor_sample}/"
                    f"{n_mols} pockets={n_pock} "
                    f"exact={_cascade_anchor_exact()}"
                )
                t_pre = time.perf_counter()
                pre_subset = self._subset_mol_dataset(
                    mol_dataset, anchor_reservoir.sample_idx
                )
                scores_pre, _ = self._score_mol_dataset(
                    {first_fold: gate_encoder},
                    {first_fold: pocket_reps_by_fold[first_fold]},
                    pre_subset,
                    use_cuda,
                    gate_bsz,
                    run_label="cascade tier 0 anchor prepass",
                )
                pre = scores_pre[first_fold]
                n_use = int(pre.shape[1])
                anchor_reservoir.buf[:, :n_use] = pre
                anchor_reservoir._pos = n_use
                medians_anchor, mads_anchor = anchor_reservoir.finalize()
                anchor_prepass_s = time.perf_counter() - t_pre
                del scores_pre
                del pre
                del pre_subset
                del anchor_reservoir
                logger.info(
                    f"cascade tier 0 anchor prepass: "
                    f"sample={anchor_sample} took {anchor_prepass_s:.2f}s"
                )
            elif do_zscore:
                # No reservoir: anchors computed after full encode (memmap/dense).
                pass

            # Allocate gate metric whenever we already have anchors (prepass) so
            # select_gather can skip the second memmap metric scan.
            if medians_anchor is not None or not do_zscore:
                gate_metric = np.empty(n_mols, dtype=np.float32)

            if use_score_memmap:
                # Default fp16 scratch (~2x less select_gather I/O); override with
                # DRUGCLIP_CASCADE_SCORE_DTYPE=float32.
                os.makedirs(score_tmp_root, exist_ok=True)
                score_tmp_dir = tempfile.mkdtemp(
                    prefix="drugclip_cascade_", dir=score_tmp_root
                )
                _, score_dev = _log_cascade_scratch(
                    "score_memmap", score_tmp_dir
                )
                score_memmap_path = os.path.join(
                    score_tmp_dir, "tier0_scores.dat"
                )
                score_memmap = np.memmap(
                    score_memmap_path,
                    dtype=score_dtype,
                    mode="w+",
                    shape=(n_pock, n_mols),
                )
                logger.info(
                    f"cascade tier 0 score memmap: shape=({n_pock}, {n_mols}), "
                    f"dtype={np.dtype(score_dtype).name}, "
                    f"~{_cascade_score_bytes(n_pock, n_mols, score_dtype) / (1024**3):.2f} GiB "
                    f"at {score_memmap_path}"
                )
            else:
                score_dev = None

            scores0, names = self._score_mol_dataset(
                {first_fold: gate_encoder},
                {first_fold: pocket_reps_by_fold[first_fold]},
                mol_dataset,
                use_cuda,
                gate_bsz,
                run_label=f"cascade tier 0 (fold {first_fold})",
                score_memmap=score_memmap,
                gate_metric=gate_metric,
                gate_medians=medians_anchor,
                gate_mads=mads_anchor,
            )
            first_scores = scores0[first_fold]  # dense or memmap (n_pock, N)
            n_pock, n_mols = first_scores.shape

            t_rank0 = time.perf_counter()
            if do_zscore and medians_anchor is None:
                # Fallback when reservoir was disabled (anchor_sample=0).
                if use_score_memmap or isinstance(first_scores, np.memmap):
                    medians_anchor, mads_anchor = _robust_pocket_anchors_memmap(
                        first_scores
                    )
                else:
                    medians_anchor, mads_anchor = _robust_pocket_anchors(
                        first_scores
                    )
                # Metric was not filled during encode without anchors; leave
                # gate_metric None so select_gather recomputes from scores.
                gate_metric = None
            anchor_s = time.perf_counter() - t_rank0

            def rank_metric(accumulated):
                # Combine the folds scored so far on the current pool with the running
                # mean ensemble, then max-over-pockets (z-scored to make pockets
                # comparable for multi-pocket targets), matching the final criterion.
                # Multi-fold: stream mean+zscore per molecule chunk (no full combined
                # matrix). Single-fold: z-score in place with a large dense chunk.
                folds = sorted(accumulated)
                dense_chunk = _dense_score_chunk_size(
                    accumulated[folds[0]].shape[0]
                )
                if do_zscore:
                    if len(folds) == 1:
                        return _max_zscore_metric(
                            accumulated[folds[0]],
                            medians_anchor,
                            mads_anchor,
                            chunk_size=dense_chunk,
                        )
                    return _max_zscore_metric_from_folds(
                        [accumulated[f] for f in folds],
                        medians_anchor,
                        mads_anchor,
                        chunk_size=dense_chunk,
                    )
                if len(folds) == 1:
                    combined = accumulated[folds[0]]
                else:
                    combined = _mean_fold_scores(
                        [accumulated[f] for f in folds]
                    )
                if combined.shape[0] == 1:
                    return combined[0]
                return np.max(combined, axis=0)

            # Tier 0: gate the full library down to pool1 using the first fold alone.
            # Prefer gather-only when gate_metric was filled during encode.
            n_pool = pool_size(tier_fracs[0], n_mols, next_pool_round_bsz(0))
            t_select0 = time.perf_counter()
            cur_idx, gathered0 = _cascade_tier0_select_gather(
                first_scores,
                medians_anchor,
                mads_anchor,
                n_pool,
                do_zscore,
                metric=gate_metric,
            )
            select_gather_s = time.perf_counter() - t_select0
            accumulated = {first_fold: gathered0}
            rank_select_s = time.perf_counter() - t_rank0
            metric_mode = (
                "precomputed" if gate_metric is not None else "memmap_scan"
            )
            logger.info(
                f"cascade tier 0 (fold {first_fold}, bsz={gate_bsz}): "
                f"pool={len(cur_idx)}/{n_mols} "
                f"(frac={cascade_frac * tier_fracs[0]:.3f})"
            )
            logger.info(
                f"cascade tier0 rank_select breakdown: "
                f"anchor_prepass={anchor_prepass_s:.2f}s "
                f"anchor={anchor_s:.2f}s select_gather={select_gather_s:.2f}s "
                f"total={anchor_prepass_s + rank_select_s:.2f}s "
                f"(zscore={do_zscore}, n_pock={n_pock}, "
                f"memmap={use_score_memmap}, metric={metric_mode}, "
                f"dtype={np.dtype(score_dtype).name if use_score_memmap else 'float32'})"
            )
            if gate_metric is not None:
                del gate_metric
                gate_metric = None

            # Drop full-library scores before tier 1 (free RAM / close memmap).
            del first_scores
            del scores0
            del gathered0
            if score_memmap is not None:
                try:
                    score_memmap.flush()
                    score_memmap._mmap.close()
                except Exception:
                    pass
                del score_memmap
                score_memmap = None
                if score_memmap_path and os.path.exists(score_memmap_path):
                    try:
                        os.remove(score_memmap_path)
                    except OSError:
                        pass
                    score_memmap_path = None
                # Finish writeback of the deleted ~GiB memmap before compact I/O.
                try:
                    os.sync()
                except (AttributeError, OSError):
                    pass
            gc.collect()

            # Immediate Subset on source — no close/compact/reload between tiers.
            t_subset0 = time.perf_counter()
            score_dataset = self._subset_mol_dataset(mol_dataset, cur_idx)
            subset_setup_s = time.perf_counter() - t_subset0

            # Tiers 1..n_gate_tiers-1: swap gate-encoder weights, score Subset,
            # and gate again on global source indices.
            for tier in range(1, n_gate_tiers):
                fold = fold_order[tier]
                t_ckpt0 = time.perf_counter()
                state = checkpoint_utils.load_checkpoint_to_cpu(ckpts[fold])
                model.load_state_dict(state["model"], strict=False)
                self._load_fold_weights_into_encoder(gate_encoder, model)
                ckpt_swap_s = time.perf_counter() - t_ckpt0
                if tier == 1:
                    logger.info(
                        f"cascade tier0→1: rank_select={rank_select_s:.2f}s "
                        f"subset_setup={subset_setup_s:.2f}s "
                        f"ckpt_swap={ckpt_swap_s:.2f}s"
                    )
                scores_f, _ = self._score_mol_dataset(
                    {fold: gate_encoder},
                    {fold: pocket_reps_by_fold[fold]},
                    score_dataset,
                    use_cuda,
                    gate_bsz,
                    run_label=f"cascade tier {tier} (fold {fold})",
                )
                accumulated[fold] = scores_f[fold]  # (n_pock, len(cur_idx))

                n_next = pool_size(
                    tier_fracs[tier], len(cur_idx), next_pool_round_bsz(tier)
                )
                t_rank_tier = time.perf_counter()
                metric = rank_metric(accumulated)
                mean_zscore_s = time.perf_counter() - t_rank_tier
                t_part = time.perf_counter()
                local = np.argpartition(metric, -n_next)[-n_next:]
                local = np.sort(local)
                argpartition_s = time.perf_counter() - t_part
                t_slice = time.perf_counter()
                cur_idx = cur_idx[local]
                # Sequential gather + free each source fold so peak stays near
                # one (n_pock, n_next) matrix instead of fancy-indexing two
                # multi-GiB C-order arrays in place.
                dense_chunk = _dense_score_chunk_size(
                    next(iter(accumulated.values())).shape[0]
                )
                new_acc = {}
                for f in list(accumulated):
                    a = accumulated.pop(f)
                    new_acc[f] = _gather_score_columns(
                        a, local, chunk_size=dense_chunk
                    )
                    del a
                accumulated = new_acc
                slice_s = time.perf_counter() - t_slice
                score_dataset = self._subset_mol_dataset(mol_dataset, cur_idx)
                rank_total_s = mean_zscore_s + argpartition_s + slice_s
                logger.info(
                    f"cascade tier {tier} (fold {fold}, bsz={gate_bsz}): "
                    f"pool={len(cur_idx)}/{n_mols} "
                    f"(frac={cascade_frac * tier_fracs[tier]:.3f})"
                )
                logger.info(
                    f"cascade tier {tier} rank_select breakdown: "
                    f"mean_zscore={mean_zscore_s:.2f}s "
                    f"argpartition={argpartition_s:.2f}s "
                    f"slice={slice_s:.2f}s total={rank_total_s:.2f}s"
                )

            # No further gate tiers: still log the tier0→1 boundary costs.
            if n_gate_tiers <= 1:
                logger.info(
                    f"cascade tier0→1: rank_select={rank_select_s:.2f}s "
                    f"subset_setup={subset_setup_s:.2f}s ckpt_swap=0.00s "
                    f"(no further gate tiers)"
                )

            del gate_encoder
            gate_encoder = None
            if use_cuda:
                torch.cuda.empty_cache()

            # Phase 2: reuse gating scores; encode remaining folds on the pool.
            pool_scores_by_fold = dict(accumulated)
            pool_dataset = score_dataset

            if do_late_compact and rescore_folds:
                # Opt-in: materialize the final (smallest) gate pool for dense
                # sequential rescore I/O. Temp LMDB under DRUGCLIP_CASCADE_TMP
                # (not necessarily the score-memmap scratch root).
                os.makedirs(compact_tmp_root, exist_ok=True)
                cascade_tmp_dir = tempfile.mkdtemp(
                    prefix="drugclip_cascade_", dir=compact_tmp_root
                )
                _, compact_dev = _log_cascade_scratch(
                    "late_compact", cascade_tmp_dir
                )
                if (
                    score_dev is not None
                    and compact_dev is not None
                    and score_dev == compact_dev
                ):
                    logger.warning(
                        "cascade scratch score_memmap and late_compact share "
                        f"st_dev={score_dev}; prefer separate NVMe roots via "
                        "DRUGCLIP_SCORE_MEMMAP_DIR and DRUGCLIP_CASCADE_TMP to "
                        "avoid writeback contention after tier0 memmap unlink"
                    )
                compact_path = os.path.join(cascade_tmp_dir, "survivors.lmdb")
                # Free the source LMDB handle before compaction: LMDB allows only
                # one open env per path per process (num_workers=0 leaves it open).
                self._close_mol_lmdb_envs(mol_dataset)
                del mol_dataset
                score_dataset = None
                t_compact0 = time.perf_counter()
                compact_lmdb_indices(
                    mol_data_path, cur_idx, compact_path, src_readahead=True
                )
                compact_copy_s = time.perf_counter() - t_compact0
                t_reload0 = time.perf_counter()
                compact_dataset = self.load_mols_dataset(
                    compact_path, "atoms", "coordinates", readahead=True
                )
                compact_reload_s = time.perf_counter() - t_reload0
                pool_dataset = compact_dataset
                logger.info(
                    f"cascade late compact before rescore: wrote {len(cur_idx)} mols "
                    f"to {compact_path} "
                    f"(compact_copy={compact_copy_s:.2f}s "
                    f"compact_reload={compact_reload_s:.2f}s)"
                )

            if rescore_folds:
                fold_encoders = {}
                for fold in rescore_folds:
                    state = checkpoint_utils.load_checkpoint_to_cpu(ckpts[fold])
                    model.load_state_dict(state["model"], strict=False)
                    fold_encoders[fold] = self._snapshot_mol_encoder(
                        model, use_cuda, use_fp16
                    )

                new_scores, _ = self._score_mol_dataset(
                    fold_encoders,
                    {f: pocket_reps_by_fold[f] for f in rescore_folds},
                    pool_dataset,
                    use_cuda,
                    rescore_bsz,
                    run_label=f"cascade rescore ({len(rescore_folds)} folds)",
                )
                pool_scores_by_fold.update(new_scores)
                logger.info(
                    f"cascade rescore ({len(rescore_folds)} folds, bsz={rescore_bsz}): "
                    f"pool={len(cur_idx)}/{n_mols} "
                    f"(reused gating folds {sorted(gate_folds)})"
                )
                fold_encoders.clear()
                if use_cuda:
                    torch.cuda.empty_cache()
            else:
                logger.info(
                    f"cascade rescore: reused all {n_folds} gating folds; "
                    f"pool={len(cur_idx)}/{n_mols}"
                )

            # Final ranking: full-mode procedure on the rescored pool.
            # Pop folds out of the dict and consume=True so each matrix is
            # released after it is added — peak RAM stays near one mean matrix
            # instead of 6 + mean (~22 GiB at 400 x 2M).
            t_rank_final = time.perf_counter()
            fold_scores = [
                pool_scores_by_fold.pop(f) for f in range(n_folds)
            ]
            res_max, rank_timings = _ensemble_rank_mols(
                fold_scores, fold_version, consume=True, return_timings=True
            )
            del fold_scores
            gc.collect()
            rank_final_s = time.perf_counter() - t_rank_final
            logger.info(
                f"cascade final rank: pool={len(cur_idx)} n_pock={n_pock} "
                f"folds={n_folds} took {rank_final_s:.2f}s "
                f"(mean={rank_timings['mean']:.2f}s "
                f"anchors={rank_timings['anchors']:.2f}s "
                f"zscore={rank_timings['zscore']:.2f}s)"
            )

            # Keep the top RETRIEVAL_TOP_FRAC relative to the FULL library size.
            k = min(len(cur_idx), max(1, int(n_mols * RETRIEVAL_TOP_FRAC)))
            if k > 0:
                top_local = np.argpartition(res_max, -k)[-k:]
                top_local = top_local[np.argsort(res_max[top_local])[::-1]]
            else:
                top_local = np.empty(0, dtype=np.int64)

            with open(save_path, "w") as f:
                for li in top_local:
                    global_i = cur_idx[li]
                    f.write(f"{names[global_i]},{res_max[li]}\n")
        finally:
            if gate_encoder is not None:
                del gate_encoder
            if score_memmap is not None:
                try:
                    score_memmap.flush()
                    mmap_obj = getattr(score_memmap, "_mmap", None)
                    if mmap_obj is not None:
                        mmap_obj.close()
                except Exception:
                    pass
                del score_memmap
            if compact_dataset is not None:
                del compact_dataset
            for tmp_dir in (score_tmp_dir, cascade_tmp_dir):
                if tmp_dir is not None:
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except OSError:
                        pass
        return

    def cascade_recall_report(
        self,
        model,
        pocket_path,
        mol_data_path,
        fold_version,
        fracs=(0.05, 0.1, 0.2),
        use_cuda=True,
        save_path=None,
        retrieval_bsz=None,
        **kwargs
    ):
        """
        For each cascade fraction, executes the true cascade retrieval on the top-N fraction
        of fold-0, including rescoring those molecules through every fold, and measures the execution time.
        Prints and optionally saves a report of recall and timings for each fraction.
        """
        ckpts = self._fold_checkpoints(fold_version)

        n_folds = len(ckpts)
        bsz = (
            retrieval_bsz
            if retrieval_bsz and retrieval_bsz > 0
            else _DEFAULT_RETRIEVAL_BSZ_CASCADE
        )
        use_fp16 = next(model.parameters()).dtype == torch.float16

        pocket_dataset = self.load_pockets_dataset(pocket_path)
        pocket_data = self._pocket_dataloader(pocket_dataset, use_cuda, batch_size=16)

        # Preload all fold pocket representations and snapshot mol encoders.
        pocket_reps_by_fold = self._cache_pocket_reps_by_fold(
            model, ckpts, pocket_data, pocket_path, fold_version, use_cuda, write_cache=True
        )
        fold_encoders = {}
        for fold in range(n_folds):
            state = checkpoint_utils.load_checkpoint_to_cpu(ckpts[fold])
            model.load_state_dict(state["model"], strict=False)
            fold_encoders[fold] = self._snapshot_mol_encoder(model, use_cuda, use_fp16)

        # Score entire library for fold-0 to determine pool gating.
        mol_dataset = self.load_mols_dataset(
            mol_data_path, "atoms", "coordinates", readahead=True
        )
        # Only need names for reporting
        scores_by_fold_0, names = self._score_mol_dataset(
            {0: fold_encoders[0]},
            {0: pocket_reps_by_fold[0]},
            mol_dataset,
            use_cuda,
            bsz,
            run_label="cascade recall gate (fold 0)",
        )
        gate = scores_by_fold_0[0].max(axis=0)
        n_pock, n_mols = scores_by_fold_0[0].shape

        # For ensemble recall, we also need the true full-library top-k reference.
        # Compute all folds ensemble on full library (analogous to old code)
        full_recall_timings = {}
        t_full0 = time.time()
        scores_by_fold_full, _ = self._score_mol_dataset(
            fold_encoders,
            pocket_reps_by_fold,
            mol_dataset,
            use_cuda,
            bsz,
            run_label="cascade recall full reference",
        )
        t_full1 = time.time()
        stack = np.stack([scores_by_fold_full[f] for f in range(n_folds)], axis=0)
        res_full_raw = np.mean(stack, axis=0)  # (n_pock, n_mols) raw
        if fold_version.startswith("6_folds") and n_pock > 1:
            # Anchor stats are computed from the RAW ensemble scores so the
            # normalization matches the full retrieval ranking exactly.
            medians = np.median(res_full_raw, axis=1, keepdims=True)
            mads = np.median(np.abs(res_full_raw - medians), axis=1, keepdims=True)
            res_full = 0.6745 * (res_full_raw - medians) / (mads + 1e-6)
        else:
            res_full = res_full_raw
        res_max_full = np.max(res_full, axis=0)  # (n_mols,)
        k = max(1, int(n_mols * RETRIEVAL_TOP_FRAC))
        full_top = np.argpartition(res_max_full, -k)[-k:]
        full_top_set = set(full_top.tolist())
        t_full = t_full1 - t_full0
        top_pct_label = f"top-{k / n_mols:.0%}"

        lines = [
            f"cascade recall report: {n_mols} mols, {n_pock} pocket(s), {n_folds} folds, full {top_pct_label} = {k}",
            f"full-library n-fold ensemble time: {t_full:.2f}s",
        ]

        for frac in fracs:
            n_pool = min(n_mols, max(1, int(n_mols * frac)))
            pool_idx = np.argpartition(gate, -n_pool)[-n_pool:]
            pool_idx_sorted = pool_idx[np.argsort(gate[pool_idx])[::-1]]
            # Use a Subset for efficient access/collate
            pool_dataset = torch.utils.data.Subset(mol_dataset, pool_idx_sorted.tolist())

            # Real cascade: run all folds on the pool. This pass exists only to
            # measure the cascade wall-clock time; its scores are intentionally
            # not used for recall (see below).
            t0 = time.time()
            self._score_mol_dataset(
                fold_encoders,
                pocket_reps_by_fold,
                pool_dataset,
                use_cuda,
                bsz
            )
            t1 = time.time()

            # Index-level recall: rank the pool by the SAME full-library scores
            # (sliced at the pool indices), so the metric is pure gate recall and
            # frac=1.0 yields 100% by construction. Restricting to a subset can
            # only lower a molecule's rank, so any full top-k member present in
            # the pool is guaranteed to remain in the pool's top-k.
            pool_scores = res_max_full[pool_idx]
            top_k = min(n_pool, k)
            if top_k > 0:
                top_local = np.argpartition(pool_scores, -top_k)[-top_k:]
            else:
                top_local = np.empty(0, dtype=np.int64)

            top_global = pool_idx[top_local]
            hit = len(full_top_set & set(top_global.tolist()))
            recall = hit / len(full_top_set) if full_top_set else float("nan")
            lines.append(
                f"  fold-0 top {frac:>6.1%} (pool={n_pool:>9}): "
                f"recall of full {top_pct_label} = {recall:7.2%} ({hit}/{len(full_top_set)}) | "
                f"cascade time = {t1 - t0:.2f}s"
            )

        report = "\n".join(lines)
        print(report)
        if save_path:
            with open(save_path, "w") as f:
                f.write(report + "\n")
        return report


