"""Retired fp16 chunk-merge path. Do not use for the quantized shard index.

This script stored shard provenance as torch.uint8 (`file_id`) and silently
wrapped past 256 embedding files. The sequential 4-bit scan in
``unimol/scan_index.py`` (then ``unimol/rerank_shortlist.py`` /
``unimol/retrieve_index.py``) supersedes it.

Kept only for reproducing old <256-file merges. Widen ``file_id`` to uint32
only if this file must stay in production use — the scan path does not need
it. Calling it with 256 or more input files now raises instead of wrapping.
"""
import os
from screening_utils import VSmodel, PocketDataset, pocket_collate_fn
import torch
import pickle
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
from itertools import repeat
from argparse import ArgumentParser


def one_process(gpu_index, mol_model, pocket_reps_path, batch_size=4, gpu_num=8, output_dir="output", file_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index + 0)
    device = torch.device("cuda")
    model = mol_model.to(device)
    model.eval()
    if isinstance(pocket_reps_path, list):
        # for npy format
        dataset = [PocketDataset(i, j) for i, j in pocket_reps_path]
        dataset = ConcatDataset(dataset)
    else:
        dataset = PocketDataset(pocket_reps_path)
    sampler = DistributedSampler(
        dataset, num_replicas=gpu_num, rank=gpu_index, shuffle=False
    )
    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=pocket_collate_fn,
        drop_last=False,
        num_workers=1,
    )
    pocket_name = []
    mol_score = []
    mol_index = []
    file_index = []

    if file_id > 0:
        with open(
            os.path.join(output_dir, f"merge{file_id-1}_{gpu_index}.pkl"),
            "rb",
        ) as f:
            pocket_name_merged, score_merged, ind_merged, f_ind_merged = pickle.load(f)
    else:
        score_merged, ind_merged, f_ind_merged = (
            repeat(None),
            repeat(None),
            repeat(None),
        )

    with torch.no_grad():
        if gpu_index == 0:
            dataLoader = tqdm(dataLoader)
        for (names, emb, index), score_m, ind_m, f_ind_m in zip(
            dataLoader, score_merged, ind_merged, f_ind_merged
        ):
            emb = emb.to(device)
            index = index.to(device)
            topk_score, topk_index = model(emb, index)
            pocket_name += names

            if file_id > 0:
                topk_score = torch.cat([score_m.to(device), topk_score], dim=1)
                topk_score, ind = torch.topk(
                    topk_score.contiguous(), k=100000, dim=-1, sorted=False
                )
                topk_index = torch.cat([ind_m.to(device), topk_index], dim=1)
                topk_index = topk_index[
                    torch.arange(topk_index.shape[0], device=device).reshape(-1, 1), ind
                ]
                # uint8 file_id wraps past 256 shards; this path is retired.
                f_ind_m = torch.cat(
                    [f_ind_m.to(device), torch.ones_like(topk_score, dtype=torch.uint8) * file_id],
                    dim=1,
                )
                f_ind_m = f_ind_m[torch.arange(f_ind_m.shape[0], device=device).reshape(-1, 1), ind]
            else:
                f_ind_m = torch.zeros_like(topk_score, dtype=torch.uint8)
            file_index.append(f_ind_m.cpu())

            mol_score.append(topk_score.cpu())
            mol_index.append(topk_index.cpu())

    if file_id > 0:
        assert pocket_name_merged == pocket_name
    with open(
        os.path.join(output_dir, f"merge{file_id}_{gpu_index}.pkl"),
        "wb",
    ) as f:
        pickle.dump((pocket_name, mol_score, mol_index, file_index), f)
    
    return None


if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    parser = ArgumentParser()
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--mol_embs", "-m", type=str, nargs="+", required=True)
    parser.add_argument("--zscore_embs", "-z", type=str, default=None)
    parser.add_argument("--pocket_reps", "-p", type=str, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--rm_intermediate", "-rm", action="store_true")
    args = parser.parse_args()

    embedding_list = args.mol_embs
    if len(embedding_list) > 256:
        raise RuntimeError(
            "utils/screening_chunk.py is retired for >=256 shards: file_id is "
            "torch.uint8 and would wrap. Use unimol/scan_index.py instead."
        )
    gpu_num = args.gpu_num
    pocket_reps = args.pocket_reps
    batch_size = args.batch_size
    zscore_embs = args.zscore_embs if args.zscore_embs else embedding_list[0]
    output_dir = args.output_dir
    rm_intermediate = args.rm_intermediate

    for i, f in tqdm(enumerate(embedding_list), total=len(embedding_list)):
        mol_model = VSmodel(f, zscore_embs)
        print(f"processing {mol_model.mol_embs.shape} mol embs from {f}")
        mp.spawn(
            one_process,
            args=(mol_model, pocket_reps, batch_size, gpu_num, output_dir, i),
            nprocs=gpu_num,
            join=True,
        )
        if rm_intermediate:
            os.system(f"rm {os.path.join(output_dir, f'merge{i-1}_*.pkl')}")
    print(f"done, final results saved at {os.path.join(output_dir, f'merge{i}_{gpu_num-1}.pkl')}")
