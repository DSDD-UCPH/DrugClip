# DrugCLIP for Drug-The-Whole-Genome

## DSDD-UCPH updates (focus: throughput + cascading)

Targeted at screening large multi-million-molecule LMDBs.

- **Faster and smaller LMDB creation:** `sdf_to_lmdb.py` builds molecule LMDBs from 3D SDFs using float16 coords with support for output ZSTD compression; entries use compound names rather than SMILES (faster and saves storage). Pass `--workers N` (or `-w N`) to parse in a process pool (default `1` is single-threaded).
- **Cascade mode (`--retrieval-mode cascade`):** for **single-target** on-the-fly screening. Cheap single-fold gates progressively shrink the library, then survivors are re-scored with all folds (same ranking as full mode). Typical **~2.5× throughput** on large libraries. See `retrieval.sh` for `CASCADE_FRAC`, `CASCADE_TIER_FRACS`, and `CASCADE_GATE_FOLDS`.
- **Faster full-mode scoring:** score aggregation and ranking were rewritten for high throughput on large libraries, with scores that are nearly identical to the original path (bonus: lower memory utilization). Caching (when enabled) is now also done accross all folds simultaneously, cutting running time when generating the cache ~3-fold.

**Full-mode fidelity: MAE **0.00403**, Pearson **0.99998**. Rankings are effectively interchangeable.

## Running on AMD GPUs

For running the DrugClip code on AMD GPUs (e.g. R9700, MI250X, MI300X) you can try the following conda setup:
```
conda create -n dsdd_drugclip python=3.10 -y
conda activate dsdd_drugclip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install faiss-gpu
pip install --no-cache-dir     iopath     lmdb     ml_collections     numpy     scipy     tensorboardX     tqdm     tokenizers h5py zstandard
cwd=`pwd` && git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core     && cd /tmp/Uni-core     && python setup.py install     && rm -rf /tmp/Uni-core && cd $cwd
git clone https://github.com/DSDD-UCPH/DrugClip.git
cd DrugClip/docker
pip3 install --no-cache-dir -r requirements.txt 
```

## Notes

- Tested: storing raw float16 in the molecule LMDB to remove the current NumPy pickle header. Raw LMDBs shrunk by ~30%, but zstd-compressed LMDBs were approximately equal in size. Not implemented.
- Workers and bsz tweaking are crucial for maximizing GPU utilization. OOTB the GPU utilization was low. In our tests (13600K with RTX 5090), a minimum of 4 workers and a block size of 384 resulted in max GPU utilization / highest throughput in full mode.


## License

This project uses different licenses for different components:

- **Source Code**: Licensed under [Apache 2.0](LICENSE)
  - The source code is freely available for both academic and commercial use.

- **Database**: Licensed under [CC BY 4.0](docs/LICENSE.md)
  - The Drug-The-Whole-Genome database is freely available for both academic and commercial use with attribution.

- **Model Weights & Outputs**: Licensed under [CC BY-NC 4.0](docs/MODEL_WEIGHTS_LICENSE.md)
  - The DrugCLIP model weights and all results generated using the model are available for non-commercial use only.
  - For commercial use, please contact the authors for licensing options.

- **MIT Licensed Components**: [MIT License](unimol/LICENSE)
  - Part of the code is modified from Uni-Mol.
  - Copyright (c) 2022 DP Technology

Please see [LICENSE.md](docs/LICENSE.md) and [MODEL_WEIGHTS_LICENSE.md](docs/MODEL_WEIGHTS_LICENSE.md) for full details.

## Model weights and encoded embeddings

link: https://huggingface.co/datasets/bgao95/DrugCLIP_data

download model_weights.zip, encoded_mol_embs.zip, targets.zip, unzip them and put them inside ./data dir


## Set environment

you can set the environment with the Dockerfile in docker dir, or use the requirements.txt file.


## Do virtual screening 

```
bash retrieval.sh
```

You need to set pocket path to ./data/targets/{target}/pocket.lmdb

target is one of the name in ./data/targets

you need to set num_folds to 8 for 5HT2A 

The molecule library for the virtual screening is 1648137 molecules inside ChemDIV.

Full mode writes two result files. `Scoring_output.txt.all_scores.txt` has every molecule in the original LMDB order:

```
index,score
```

`Scoring_output.txt` has the top 100000 molecules, highest score first. The name is the library entry (compound name or SMILES):

```
index,name,score
```

### For data won't fit in mem

To perform screening on chucked mol embedding files, run:

```shell
python utils/screening_chucks.py --gpu_num 8 --mol_embs <path_to_multiple_chucks> --zscore_embs <uniform_small_set_for_approx_zscore> --pocket_reps <path_to_pocket_reps_pkl> --batch_size <batch_size> --output_dir <path_to_output_dir> --rm_intermediate
```

The `--mol_embs` argument allows multiple args. Each arg should be a path to a chunk of molecule embeddings. The `--zscore_embs` argument should be a path to a small set of molecule embeddings for approximating the zscore; if not specified, the first `mol_embs` file is used. The resulting files will be saved as `merge{mol_embs_file_id}_{gpu_id}.pkl` in `output_dir` . The `--rm_intermediate` flag will remove the intermediate files after the next chunk for saving disk space.

To retrieve SMILES strings and original ids for the output files, run:

```shell
python utils/retrieve_chunk.py --input_files output/merge*.pkl --mol_lmdb <path_to_mol_lmdb> --output_dir retrieval_results --num_threads <num_threads>
```

The `--mol_lmdb` should be extractly the same order as the `--mol_embs` argument in the last step. The resulting files will be saved as `retrieval_results/{pocket_name}.csv`.

## Benchmarking

link: https://huggingface.co/datasets/bgao95/DrugCLIP_data

download DUD-E.zip, LIT-PCBA.zip, unzip them and put inside ./data dir


```
bash test.sh
```

select TASK to DUDE or PCBA in test.sh


## Other tools

Pocket Pretraining: https://github.com/THU-ATOM/ProFSA

virtual screening post-processing: https://github.com/THU-ATOM/DrugCLIP_screen_pipeline

Pocket detection: https://github.com/THU-ATOM/Pocket-Detection-of-DTWG







