# Revisiting SASRec on Amazon Reviews 2023

[![CI](https://github.com/alina0607/sasrec-amazon-2023/actions/workflows/ci.yml/badge.svg)](https://github.com/alina0607/sasrec-amazon-2023/actions/workflows/ci.yml)

A reproducibility study of **SASRec** (Self-Attentive Sequential Recommendation; Kang & McAuley, ICDM 2018) on the 2023 release of the Amazon review data.

> **Status:** work in progress. The whole project is one notebook, [`sasrec_amazon_2023.ipynb`](sasrec_amazon_2023.ipynb); sections marked TODO are not written yet.

## Objective

The original paper evaluated SASRec on an earlier version of the Amazon dataset. Since then, the data has been re-collected with a longer time span and a revised category structure. We re-run the paper's experimental protocol on [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and check whether its conclusions still hold.

## Research questions

- **RQ1 (main question).** Do the sampled evaluation used in the paper (ranking the ground-truth item against 100 sampled negatives) and full ranking over all items put the methods in the same order?
- **RQ2.** Does SASRec still rank above the non-neural baselines on Amazon Reviews 2023? We compare the order of the methods with the paper's, not the exact numbers, because the data differs.

We start with SASRec, PopRec and BPR under both protocols; FMC and FPMC follow as time allows.

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
| `docs/team_spec.md` | how the four of us work on one notebook |
| `data/` | the Amazon Reviews 2023 files, downloaded here by the notebook on its first run (not in git) |
| `results/` | small CSV files with the trained results, read when `RUN_EXPERIMENTS = False` |

## Data

We use one category of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/),
**`Industrial_and_Scientific`**, chosen because it is small enough to train every model several
times on a laptop (the teacher's feedback on our proposal). We read the official 5-core
`last_out_w_his` files, which the dataset's authors already split with the leave-one-out
strategy. After loading:

| Users | Items | Interactions |
|---:|---:|---:|
| 50,985 | 25,848 | 412,947 |

These match the official 5-core table (51.0K / 25.8K / 412.9K), and every user's validation and
test input matches the official `history` column. The first run of the notebook downloads the
three files (about 14 MB) into `data/Industrial_and_Scientific/`; later runs read them from
there. The dataset is released by the McAuley Lab for research use; this project uses it for
academic purposes only.

## Running the notebook

Python 3.11 or newer, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # installs the packages from uv.lock into .venv
uv run jupyter lab         # or open the notebook in VS Code with the .venv kernel
```

Run all cells from the top; the first run needs an internet connection to download the data.
If you commit to this repository, run `uv run pre-commit install` once: it clears notebook
outputs and formats the code on every commit. GitHub Actions lints the notebook and runs it from top to bottom
on every push and pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). On
Linux the CPU build of PyTorch is installed, so CI does not download the CUDA libraries.

## References

[1] W.-C. Kang and J. McAuley, "Self-attentive sequential recommendation," in *Proc. IEEE ICDM*, 2018. [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)

[2] Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley, "Bridging language and items for retrieval and recommendation," arXiv:2403.03952, 2024. [arXiv:2403.03952](https://arxiv.org/abs/2403.03952)

[3] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback," in *Proc. UAI*, 2009.

[4] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, "Factorizing personalized Markov chains for next-basket recommendation," in *Proc. WWW*, 2010.

[5] W. Krichene and S. Rendle, "On sampled metrics for item recommendation," in *Proc. ACM SIGKDD*, 2020.

The full list, in the format we hand in, is in Section 11 of the notebook.
