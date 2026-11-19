# Revisiting SASRec on Amazon Reviews 2023

[![CI](https://github.com/alina0607/sasrec-amazon-2023/actions/workflows/ci.yml/badge.svg)](https://github.com/alina0607/sasrec-amazon-2023/actions/workflows/ci.yml)

A reproducibility study of **SASRec** (Self-Attentive Sequential Recommendation; Kang & McAuley, ICDM 2018) on the 2023 release of the Amazon review data.

> **Status:** work in progress. The whole project is one notebook, [`sasrec_amazon_2023.ipynb`](sasrec_amazon_2023.ipynb); sections marked TODO are not written yet.

## Objective

The original paper evaluated SASRec on an earlier version of the Amazon dataset. Since then, the data has been re-collected with a longer time span and a revised category structure. We re-run the paper's experimental protocol on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and check whether its conclusions still hold.

## Research questions

- **RQ1.** Does SASRec still outperform the non-neural baselines (PopRec, BPR, FMC, FPMC) on Amazon Reviews 2023?
- **RQ2.** Which components of SASRec matter most? We focus on positional embeddings, shared item embeddings, and the number of self-attention blocks.
- **RQ3.** Do the conclusions change when the sampled evaluation used in the paper (ranking the ground-truth item against 100 sampled negatives) is replaced with full ranking over all items?

## Protocol

Following the original paper:

- Every review is treated as implicit feedback, and we use the 5-core filtered data.
- Leave-one-out split: the most recent interaction is used for testing, the second most recent for validation.
- Metrics: Hit@10 and NDCG@10.
- Hyperparameters are tuned on the validation set only, and each trained model is run with three random seeds.

## Repository layout

| Path | Contents |
|---|---|
| `sasrec_amazon_2023.ipynb` | the project: data, models, training, evaluation, results and discussion |
| `docs/team_spec.md` | how the four of us work on one notebook (git, branches, checks) |
| `data/` | the Amazon Reviews 2023 files the notebook reads |
| `results/` | small CSV files with the trained results, read when `RUN_EXPERIMENTS = False` |

## Data

The notebook uses the official 5-core `last_out_w_his` files of
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), stored in `data/` so the
notebook runs after a `git pull` with no download. The dataset is released by the McAuley
Lab for research use; this project uses it for academic purposes only.

## Running the notebook

Python 3.11 or newer, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # installs the packages from uv.lock into .venv
uv run pre-commit install  # once: clears notebook outputs and formats code on commit
uv run jupyter lab         # or open the notebook in VS Code with the .venv kernel
```

Run all cells from the top. GitHub Actions lints the notebook and runs it from top to bottom
on every push and pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). On
Linux the CPU build of PyTorch is installed, so CI does not download the CUDA libraries.

## References

- Wang-Cheng Kang and Julian McAuley. *Self-Attentive Sequential Recommendation.* ICDM 2018. [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
- Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. *Bridging Language and Items for Retrieval and Recommendation.* arXiv preprint, 2024. [arXiv:2403.03952](https://arxiv.org/abs/2403.03952)
