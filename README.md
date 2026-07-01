# CombiSearch

Code and configurations for *Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples* (ACL 2025, <https://aclanthology.org/2025.acl-long.1393.pdf>).

## Requirements

- Python 3.10–3.12, CUDA 12.4.
- A CUDA GPU for the Llama tables (8B / 70B); the 70B uses the 4-bit GPTQ checkpoint
  [`holi-lab/Meta-Llama-3-70B-Instruct-GPTQ`](https://huggingface.co/holi-lab/Meta-Llama-3-70B-Instruct-GPTQ).
- An OpenAI API key for the gpt-3.5-turbo table (Table 1).

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e '.[local-llm]'
```

## Data

Download and build every dataset into `data/`:

```bash
bash runner/preprocess.sh
```

Point to a local SGD corpus
(else the SGD step is skipped):

```bash
SGD_DATA_DIR=/path/to/SGD bash runner/preprocess.sh
```

## Retrievers and indexes

The default (non-`FULL`) table runs expect the trained retrievers and prebuilt SBERT
indexes under `outputs/`. Download the retrievers from Hugging Face:

```bash
huggingface-cli download holi-lab/combisearch-retrievers --repo-type dataset --local-dir outputs
```

This restores the layout the configs reference:

```
outputs/retrievers/finetuning/mwoz21/{combisearch,refpydst,individual}/<pct>/split_v*
outputs/retrievers/finetuning/sgd/{combisearch,refpydst}
outputs/retrievers/sentence-transformers_all-mpnet-base-v2/pretrained
outputs/indexes/sentence-transformers_all-mpnet-base-v2/<input_type>/train_index.npy
```

Alternatively, rebuild everything from scratch by running any table with `FULL=1`.

## Reproducing a table

Each table has one runner. By default it runs only that table's stage, using the
artifacts already in `outputs/`:

```bash
bash runner/table2.sh
```

`FULL=1` rebuilds the upstream artifacts first (preprocess + indexes, plus data generation
and retriever fine-tuning for the trained-retriever tables). `DRY_RUN=1` prints the commands
without running them. Retriever fine-tuning uses torchrun across `NUM_GPUS` GPUs (default 4);
set `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` to match your machine.

```bash
FULL=1 bash runner/table2.sh                 # rebuild upstream, then run
DRY_RUN=1 bash runner/table2.sh              # print only
OPENAI_API_KEY=sk-... bash runner/table1.sh  # gpt-3.5-turbo (Table 1)
```

| Runner | Dataset · DST model | Measures | Main stage |
| --- | --- | --- | --- |
| `table1.sh` | MultiWOZ 2.1/2.4 · gpt-3.5-turbo | CombiSearch vs RefPyDST | DST inference (S4) |
| `table2.sh` | MultiWOZ 2.4 · Llama-3 8B + 70B | CombiSearch vs RefPyDST | trained-retriever inference (S4) |
| `table3.sh` | SGD (+ MultiWOZ) · Llama-3 8B | Random / RefPyDST / CombiSearch | trained-retriever inference (S4) |
| `table4.sh` | MultiWOZ coref · Llama-3 8B | JSON vs Python prompt format | trained-retriever inference (S4) |
| `table8.sh` | MultiWOZ 2.4 · Llama-3 8B + 70B | CombiSearch vs Individual scoring | trained-retriever inference (S4) |
| `table5.sh` | MultiWOZ · Llama-3 8B + 70B | oracle upper bound (RefPyDST / Hybrid / CombiSearch) | combinatorial scoring (S2) + baselines (S4) |
| `table6.sh` | MultiWOZ · Llama-3 8B | oracle, pool construction | combinatorial scoring (S2) + baselines (S4) |
| `table7.sh` | MultiWOZ · Llama-3 8B | oracle, individual vs CombiSearch (M=3/9) | combinatorial scoring (S2) |

Results are written to `outputs/runs/<table>/**/running_log.json`. Score one saved log:

```bash
.venv/bin/python -m combisearch.evaluation.score_run \
  --running_log outputs/runs/table_2/.../running_log.json \
  --test_fn data/mw24_100p_test.json
```

Or aggregate every run of a table into per-config JGA means (over `split_v*` seeds),
printed as a table and optionally written to CSV:

```bash
.venv/bin/python -m combisearch.evaluation.summarize --table table_2 --out table_2.csv
```

## Running a single config

Every stage is config-driven and can be invoked directly on one config file:

```bash
combisearch-run-dst        configs/table_2/8B/combisearch/5p/split_v1.json
combisearch-generate-data  configs/data_generation/mwoz21/5p/split_v1.json
combisearch-train-retriever configs/finetuning/mwoz21/combisearch/5p/split_v1.json
combisearch-build-indexes
```

`combisearch-check-artifacts` reports which prior-stage inputs
referenced by the configs are missing.

## Repository layout

```
src/combisearch/   package: retrieval, prompting, DST pipeline, retriever training, CLI
configs/           one JSON per run, grouped by table plus data_generation/ and finetuning/
runner/            tableN.sh reproduction scripts, preprocess.sh, and stages/ (s0–s4)
data/              datasets after preprocessing, plus data/code/ preprocessing scripts
outputs/           trained retrievers, indexes, and run logs (git-ignored)
```

## Acknowledgements

A substantial portion of this codebase is adapted from **RefPyDST** (King & Flanigan,
"Diverse Retrieval-Augmented In-Context Learning for Dialogue State Tracking", Findings of
ACL 2023, <https://github.com/jlab-nlp/RefPyDST>), which itself builds on
**IC-DST** (Hu et al., 2022,
<https://github.com/Yushi-Hu/IC-DST>). The MultiWOZ text-normalization code derives from the
original MultiWOZ repository (<https://github.com/budzianowski/multiwoz>).

```bibtex
@inproceedings{king-flanigan-2023-diverse,
    title = "Diverse Retrieval-Augmented In-Context Learning for Dialogue State Tracking",
    author = "King, Brendan  and
      Flanigan, Jeffrey",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.344",
    doi = "10.18653/v1/2023.findings-acl.344",
    pages = "5570--5585"
}

@article{hu2022context,
  title={In-Context Learning for Few-Shot Dialogue State Tracking},
  author={Hu, Yushi and Lee, Chia-Hsuan and Xie, Tianbao and Yu, Tao and Smith, Noah A and Ostendorf, Mari},
  journal={arXiv preprint arXiv:2203.08568},
  year={2022}
}
```

```bibtex
@inproceedings{pyun-etal-2025-improving,
    title = "Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples",
    author = "Pyun, Haesung  and
      Park, Yoonah  and
      Jo, Yohan",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1393/",
    doi = "10.18653/v1/2025.acl-long.1393",
    pages = "28694--28714",
    ISBN = "979-8-89176-251-0",
}
```
