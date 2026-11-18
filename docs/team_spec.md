# Code specification and contributor guide

This document fixes how the code of the project is organised and tells each team member
how to write their part. It is the technical companion of the team plan: the plan says
*what* each person does and *when*; this document says *how the code must look* so that
all parts fit together.

**Read first:** everyone reads Sections 1 to 5. Then:

| You are | Also read |
|---|---|
| Frank (shared data loader, evaluation, PopRec, Random, results table) | 6, 7, 8.3, 9, 10 |
| Kyle (BPR, FMC, FPMC; proposal and related work are outside this document) | 8, 9, 10, 11 |
| Matthew (dataset checks, figures, notebook integration) | 6.4, 12 |
| Alina (SASRec and ablations, GRU4Rec, shared code) | everything |

---

## 1. Fixed decisions

These values are used by every script. They live in one file,
`src/sasrec2023/constants.py`. Do not copy the numbers into your own code; import them.

| Decision | Value | Why |
|---|---|---|
| Data | Amazon Reviews 2023, 5-core, `last_out_w_his` split | Matches the paper's leave-one-out protocol (Section IV-A) |
| Evaluation sampling seed | `EVAL_SEED = 0` | One fixed set of 100 negatives for every method |
| Training seeds | `TRAIN_SEEDS = (1, 2, 3)` | Team plan 3.3 rule 4 |
| Negatives per case | `N_NEGATIVES = 100` | Paper Section IV-D |
| Cutoff | `K = 10` (Hit@10, NDCG@10) | Paper Section IV-D |
| Full-ranking list length | `TOP_N = 50` | Team plan 3.4 |
| Early stopping | stop when validation NDCG@10 (sampled) has not improved for `PATIENCE = 20` epochs; at most `MAX_EPOCHS = 200` | Paper Section IV-C; the cap keeps runs finite |
| Metric that selects epochs and hyperparameters | **sampled** validation NDCG@10 | The protocol of the paper; full ranking is reported, not used for selection |
| Ties in the sampled ranking | count against the answer | A model that scores every item equally must get rank 101, not rank 1 |
| History-length groups | `1–5`, `6–10`, `>10` items of input history | See the note below |
| User and item ids in all files | integer indices from the mapping files (Section 6.3) | Every file joins on the same ids |

**Why the history groups are not `≤3 / 4–10 / >10`.** In 5-core data every user has at
least 5 interactions, so a test case always has at least 4 items of input history and a
`≤3` group is empty. Measured on the 2014 Beauty 5-core data: 0 of 22,363 test cases
have 3 or fewer items. The groups are a constant (`HISTORY_GROUPS`), so Matthew can propose
other limits after looking at the 2023 data.

---

## 2. Repository layout and owners

```
sasrec-amazon-2023/
├── src/sasrec2023/
│   ├── constants.py        Alina   fixed numbers (Section 1)
│   ├── config.py           Alina   ModelConfig base class
│   ├── registry.py         Alina   metaclass that checks and registers models
│   ├── base.py             Alina   ScoringRecommender base class, EarlyStopping
│   ├── decorators.py       Alina   seeded (fixes all random generators for a block)
│   ├── export.py           Alina   writes the two result files (Section 9)
│   ├── data.py             both    Split: Alina; load_data: Frank (Section 6)
│   ├── candidates.py       Frank   makes and reads candidate files (Section 7)
│   ├── evaluate.py         Frank   metrics (Section 10)
│   └── models/
│       ├── __init__.py     Alina   imports every model so it registers
│       ├── poprec.py       Frank
│       ├── random_rec.py   Frank   (not random.py: that name hides Python's random module)
│       ├── bpr.py          Kyle
│       ├── fmc.py          Kyle
│       ├── fpmc.py         Kyle
│       ├── sasrec.py       Alina
│       └── gru4rec.py      Alina
├── scripts/
│   ├── prepare_data.py     Frank   downloads, checks, writes mapping and candidate files
│   ├── run.py              Alina   trains one model and writes its result files
│   └── results_table.py    Frank   reads result files and prints the results table
├── tests/                  everyone: one test file per module
├── notebooks/              Matthew (EDA), Frank (results analysis), Alina (ablation analysis)
├── data/                   not in git (Section 6)
└── results/                not in git (Section 9)
```

Only the owner edits a file. If you need a change in someone else's file, ask them or
open a pull request and ask them to review it.

---

## 3. The idea in one picture

```
 official csv.gz files
        │  load_data (Frank)
        ▼
      Split ──────────────► model.fit(split)          (Kyle, Alina)
        │                         │
        │                         ▼
        │                 model.score_all(users, histories)
        │                         │
        │  candidates (Frank)     │ export (Alina)
        └────────────────►────────┴──► results/<method>_<split>_seed<s>_sampled.csv.gz
                                  └──► results/<method>_<split>_seed<s>_top50.csv.gz
                                                │ evaluate (Frank)
                                                ▼
                                       Hit@10, NDCG@10, per group
```

Every method writes the same two files in the same way, because the writing is done once,
in `export.py`, for all methods. A model author never writes result files by hand and
never computes metrics by hand.

---

## 4. What every model author must know about the classes

Every method in the study (PopRec, Random, BPR, FMC, FPMC, SASRec, GRU4Rec) is a subclass
of `ScoringRecommender`. A model is "something that gives every item a score; a higher
score means recommend it first". That is all the base class assumes.

To write a model you do **three things** and write **two methods**:

| You must | Example |
|---|---|
| set the class attribute `name` | `name = "bpr"` |
| set the class attribute `config_cls` | `config_cls = BPRConfig` |
| set `self.config` in `__init__` | `self.config = config or BPRConfig()` |
| write `fit(self, split)` | train the model |
| write `score_all(self, users, histories)` | return a `(len(users), n_items)` tensor |

You get these for free from the base class; do not write them yourself:

| Inherited | What it does |
|---|---|
| `recommend(users, histories, k)` | top `k` items per case, items already in the history removed |
| `score_candidates(users, histories, candidates)` | the scores of the 101 candidates only |
| `method` | the name used in result files: `bpr`, or `sasrec_nope` for an ablation variant |

### 4.1 The base class

```python
class ScoringRecommender(metaclass=RecommenderMeta):
    """A model that gives every item a score; higher means recommend first."""

    name: ClassVar[str]
    config_cls: ClassVar[type[ModelConfig]]
    config: ModelConfig

    @abstractmethod
    def fit(self, split: Split) -> None:
        """Train. May read `split.train()` and `split.valid()`; never `split.test()`."""

    @abstractmethod
    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        """Score every item for every case: shape `(len(users), n_items)`."""
```

**Why `score_all` gets both `users` and `histories`.** BPR and FPMC look up a vector for
the user, so they need the user index. FMC, SASRec and GRU4Rec read the history. PopRec
and Random need neither. Passing both lets every model take what it needs from one
signature. Case `r` is user `users[r]` with input history `histories[r]`.

### 4.2 The metaclass: what it checks for you

You never use the metaclass directly. It runs when Python reads your `class` line and
when someone creates your model, and it stops common mistakes early with a clear message:

| Mistake | When it fails | Message |
|---|---|---|
| forgot `name` | when the module is imported | `BPR must set a class attribute 'name'` |
| `config_cls` is not a `ModelConfig` subclass | when the module is imported | `BPR must set 'config_cls' to a ModelConfig subclass` |
| two models with the same `name` | when the module is imported | `name 'bpr' is already used` |
| `__init__` does not set `self.config` | when the model is created | `BPR.__init__ must set self.config` |
| forgot `fit` or `score_all` | when the model is created | Python's `Can't instantiate abstract class ...` |

It also registers every model under its `name`, so `scripts/run.py --model bpr` finds
your class without anyone editing the script. For this to work, add your module to the
import line in `src/sasrec2023/models/__init__.py`.

### 4.3 Configurations

Every model has a configuration: a frozen dataclass that holds every number that changes
the result. "Frozen" means it cannot be changed after it is built, so the values recorded
for a run are exactly the values the model used.

```python
@dataclass(frozen=True)
class ModelConfig:
    seed: int = 1  # seeds every random number generator during fit
    variant: str = ""  # ablation name; result files use "<name>_<variant>"


@dataclass(frozen=True)
class BPRConfig(ModelConfig):
    d: int = 50  # embedding size, tuned over {10, 20, 30, 40, 50}
    l2: float = 0.001  # L2 weight, tuned over {0.0001, 0.001, 0.01, 0.1, 1}
    lr: float = 0.001
    batch_size: int = 1024
```

Rules:

- Every hyperparameter is a field of the config. No numbers hidden inside `fit`.
- Defaults are the values the paper states. If the paper does not state a value, the
  field has a comment that says so.
- Ablation variants are configurations, not new classes:
  `SASRecConfig(use_pe=False, variant="nope")`.

### 4.4 Worked example: PopRec

The whole model, as it should look. Use it as your template.

```python
"""PopRec: recommend the items that appear most often in the training sequences."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sasrec2023.base import ScoringRecommender
from sasrec2023.config import ModelConfig
from sasrec2023.data import History, Split


@dataclass(frozen=True)
class PopRecConfig(ModelConfig):
    """PopRec has nothing to tune."""


class PopRec(ScoringRecommender):
    name = "poprec"
    config_cls = PopRecConfig

    def __init__(self, config: PopRecConfig | None = None) -> None:
        self.config = config or PopRecConfig()
        self._scores: torch.Tensor | None = None

    def fit(self, split: Split) -> None:
        counts = Counter(item for seq in split.train() for item in seq)
        # Ties go to the lower item index, so every score is distinct.
        order = sorted(range(split.n_items), key=lambda i: (-counts[i], i))
        scores = torch.empty(split.n_items)
        scores[order] = torch.arange(split.n_items - 1, -1, -1, dtype=torch.float32)
        self._scores = scores

    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        if self._scores is None:
            raise RuntimeError("call fit() before scoring")
        return self._scores.expand(len(users), -1)
```

---

## 5. Rules for every model

1. **`fit` reads `split.train()` and `split.valid()` only.** Reading `split.test()` in
   `fit` is a leak. The contract test (Section 11) checks this and fails if you do.
2. **Seeds.** Everything random in `fit` (initial weights, negative sampling, batch order)
   must come from `self.config.seed`. Use `with seeded(self.config.seed):` around the
   training (from `sasrec2023.decorators`). Two fits with the same config must give the
   same scores; the contract test checks this too.
3. **No result files, no metrics in your model file.** `export.py` writes the files and
   A's `evaluate.py` computes the metrics.
4. **Item index `n_items` does not exist.** Scores have exactly `n_items` columns. If your
   model needs a padding item (SASRec, GRU4Rec), give it index `n_items` inside the model
   and drop that column before returning from `score_all`.
5. **Early stopping uses the shared helper** (Section 8.2), so every trained model stops
   by the same rule.
6. **Device.** Default to CPU. A `device` argument in `__init__` is allowed; return the
   scores on CPU.

---

## 6. Data (Frank)

### 6.1 Source files

From the Amazon Reviews 2023 benchmark page (https://amazon-reviews-2023.github.io/,
"Pure IDs (5-Core)"), leave-last-out split:

```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/<Category>.train.csv.gz
                                                                                   ....valid.csv.gz
                                                                                   ....test.csv.gz
```

The page shows this header for the test file:
`user_id,parent_asin,rating,timestamp,history`.
The page does **not** state:

- whether the train and valid files have the same columns;
- whether the `history` of a test row includes the validation item;
- the unit of `timestamp` (the example has 13 digits, which suggests milliseconds);
- the exact 5-core rule.

So `load_data` must **check** these instead of assuming them (Section 6.2), and the
answers go into the report's data section.

### 6.2 `load_data`

```python
def load_data(category: str, data_dir: Path = Path("data")) -> Split:
    """Read the official last_out_w_his files of one category into a Split."""
```

How to build each user's sequence:

1. Training items: the user's rows in `train`, sorted by `timestamp` with a **stable**
   sort (rows with the same timestamp keep their file order).
2. Then the user's single `valid` item, then the single `test` item.
3. Convert `parent_asin` and `user_id` to integer indices with the mapping files
   (Section 6.3).

Checks that must raise an error with a clear message if they fail (not print a warning):

| Check | Why |
|---|---|
| every user has exactly one valid row and one test row | leave-one-out |
| every user has at least one training row | `Split` needs 3 items per user |
| the valid row's `history` equals the user's training items, in order | proves step 1 gives the official order |
| the test row's `history` equals the valid row's `history` + the valid item, **or** equals the valid row's `history` | tells us which of the two the official files do; write down which one |
| no item of valid/test is missing from the item mapping | ids are consistent |

Count and report (do not fail) how often the same user–item pair appears more than once
in a sequence. Matthew reports the same number in the EDA; both must agree.

### 6.3 Mapping files

`scripts/prepare_data.py` writes, once per category:

```
data/<Category>/item_map.csv     index,parent_asin     index = 0, 1, 2, ...
data/<Category>/user_map.csv     index,user_id
```

Indices are given in order of first appearance when the train file is read top to
bottom. The files are written once and never regenerated; everyone uses the same copy.
`prepare_data.py` prints the SHA-256 of both files; put the hashes in the README so
everyone can check they have the same files.

### 6.4 The `Split` object

`load_data` returns this. Every model, script and notebook reads the data only through it.
Alina writes this class first (step 1 of Section 14), so models and tests can use it
before the real data is ready; Frank then adds `load_data` to the same file.

```python
History = tuple[int, ...]


@dataclass(frozen=True)
class Split:
    """Every user's item sequence, oldest first, cut by leave-one-out.

    User index `u` is position `u` in `sequences` and `users`.
    Item index `j` is position `j` in `items` and column `j` of a score matrix.
    """

    sequences: tuple[History, ...]  # training items, then valid item, then test item
    users: tuple[str, ...]  # original user_id of each user index
    items: tuple[str, ...]  # original parent_asin of each item index

    @property
    def n_users(self) -> int: ...
    @property
    def n_items(self) -> int: ...

    def train(self) -> list[History]:
        """Training sequences: everything but the last two items."""
        return [s[:-2] for s in self.sequences]

    def valid(self) -> tuple[list[History], list[int]]:
        """Validation cases: (training sequence, second-to-last item)."""
        return [s[:-2] for s in self.sequences], [s[-2] for s in self.sequences]

    def test(self) -> tuple[list[History], list[int]]:
        """Test cases: (training sequence + validation item, last item)."""
        return [s[:-1] for s in self.sequences], [s[-1] for s in self.sequences]
```

The test input contains the validation item, as the paper says (Section IV-A). For both
`valid()` and `test()`, case `u` belongs to user `u`, so the `users` argument of
`score_all` is simply `range(split.n_users)`.

---

## 7. Candidate files (Frank)

```
data/<Category>/candidates_valid.csv.gz
data/<Category>/candidates_test.csv.gz
columns: user_id,item_id,is_answer       101 rows per user, exactly one with is_answer = 1
```

How the 100 negatives are drawn:

- uniformly at random from the items the user has **never** interacted with, where
  "interacted" means every item in the user's whole sequence (train, valid and test);
- with `random.Random(EVAL_SEED)`, users in index order, so the files can be rebuilt
  exactly;
- without repeats within one user.

`candidates.py` also provides

```python
def read_candidates(category: str, part: Literal["valid", "test"]) -> torch.Tensor:
    """(n_users, 101) item indices; row u is user u, column 0 is the answer."""
```

This tensor is what models and `export.py` use. Putting the answer in column 0 lets the
metric code find it without a lookup. `prepare_data.py` prints the SHA-256 of both
candidate files; the hashes go in the README.

---

## 8. Training rules (Kyle, Alina)

### 8.1 Hyperparameter search

| Model | Tuned | Grid |
|---|---|---|
| BPR, FMC, FPMC | `d`, `l2` | `d` ∈ {10, 20, 30, 40, 50}; `l2` ∈ {0.0001, 0.001, 0.01, 0.1, 1} |
| SASRec | `d` | {10, 20, 30, 40, 50}; everything else fixed at the paper's values |
| GRU4Rec | `d` | {10, 20, 30, 40, 50} |

Procedure:

1. Run every grid point with seed 1, on validation only.
2. Pick the point with the highest sampled validation NDCG@10. If two points tie to four
   decimals, take the smaller `d`, then the smaller `l2`.
3. Run the chosen point with seeds 1, 2 and 3. Write valid and test result files.
4. Test is run **once**, in step 3. `scripts/run.py` refuses to overwrite an existing
   test result file, so a second look at test is not possible by accident.

Keep the full grid table (one row per point, its validation NDCG@10). It goes into the
report to show the baselines were tuned (team plan 3.3 rule 3).

### 8.2 Early stopping

After every epoch, score the validation candidates and compute sampled NDCG@10 with
A's `evaluate` functions. Use the shared helper so all models stop by the same rule:

```python
stopper = EarlyStopping(patience=PATIENCE)      # from sasrec2023.base
for epoch in range(1, MAX_EPOCHS + 1):
    ...train one epoch...
    ndcg = validation_ndcg(self, split, candidates)   # sampled NDCG@10
    if stopper.step(ndcg, self.state_dict()):         # True = stop now
        break
self.load_state_dict(stopper.best_state)               # keep the best epoch
```

Record whether a run stopped early or hit `MAX_EPOCHS`; `run.py` logs it. If a model
hits the cap on its chosen configuration, tell Alina: the cap may need to go up for
everyone.

### 8.3 PopRec and Random (Frank)

- **PopRec:** counts from `split.train()` only, the same scores for valid and test
  (Section 4.4). It has no randomness; `run.py` still writes one file per training seed
  (identical files) so the results table needs no special case.
- **Random:** `score_all` returns uniform random numbers from a generator seeded with
  `self.config.seed`. Expected sampled Hit@10 is 10 / 101 ≈ 0.099. Measured with the
  metric code of Section 10 on 20,000 random cases, the result is within 0.01 of that.

---

## 9. Result files (Alina's `export.py`)

`run.py` calls `export` after `fit`. Model authors do nothing for this step.

```
results/<method>_<split>_seed<seed>_sampled.csv.gz   user_id,item_id,score    101 rows per user
results/<method>_<split>_seed<seed>_top50.csv.gz     user_id,rank,item_id     50 rows per user, rank 1 = best
results/<method>_<split>_seed<seed>.json             config, epochs trained, stopped early or not, run time
```

- `sampled`: `score_candidates(users, histories, candidates)` for the 101 candidates.
- `top50`: `recommend(users, histories, 50)`, which removes items already in the input
  history.
- Files are gzip-compressed (`.csv.gz`) because a large category has millions of rows;
  `pandas.read_csv` reads them directly.
- Tuning runs (Section 8.1 step 1) go to `results/tuning/<method>/d30_l2-0.001/...` so
  they never overwrite the final files.

Result files are too large for git. The team shares the `results/` folder through one
shared cloud folder (link in the README). The `.json` summaries are small and may be
committed.

Method names: `poprec`, `random`, `bpr`, `fmc`, `fpmc`, `sasrec`, `gru4rec`.
Ablation variants (Alina):

| Variant | Config | File name |
|---|---|---|
| no positional embedding | `use_pe=False` | `sasrec_nope` |
| unshared item embedding | `share_item_emb=False` | `sasrec_unshared` |
| 0 / 1 / 3 blocks | `blocks=0`, `1`, `3` | `sasrec_blocks0`, `sasrec_blocks1`, `sasrec_blocks3` |
| no residual connections | `use_residual=False` | `sasrec_nores` |
| no dropout | `dropout=0.0` | `sasrec_nodrop` |
| 2 heads | `heads=2` | `sasrec_heads2` |

---

## 10. Evaluation (Frank)

`evaluate.py` has two layers. The lower layer is shared by the in-memory path (early
stopping during training) and the file path (final numbers), so both always give the
same number for the same scores.

```python
def ranks_from_candidate_scores(scores: torch.Tensor) -> list[int]:
    """Rank of the answer among 101 candidates; column 0 is the answer.

    Ties count against the answer, so a model that scores everything equally
    gets rank 101, not rank 1.
    """
    answer = scores[:, :1]
    return (1 + (scores[:, 1:] >= answer).sum(dim=1)).tolist()


def metrics_from_ranks(ranks: Sequence[int | None], k: int = K) -> dict[str, float]:
    """Hit@k and NDCG@k; `None` means the answer is not in the list."""
    hit = [1.0 if r is not None and r <= k else 0.0 for r in ranks]
    ndcg = [1 / math.log2(r + 1) if r is not None and r <= k else 0.0 for r in ranks]
    return {f"hit@{k}": sum(hit) / len(hit), f"ndcg@{k}": sum(ndcg) / len(ndcg)}


def history_group(length: int) -> str:
    """Name of the group a case falls in, by the length of its input history."""
```

The upper layer Frank writes on top:

```python
def evaluate_sampled(path: Path, category: str, part: str) -> dict[str, float]:
    """Read a *_sampled.csv.gz file, join it with the candidate file, compute metrics."""

def evaluate_full(path: Path, split: Split, part: str) -> dict[str, float]:
    """Read a *_top50.csv.gz file; rank = position of the answer, None if absent."""

def evaluate_by_group(..., split: Split, part: str) -> dict[str, dict[str, float]]:
    """The same metrics per history_group(len(input history))."""
```

`evaluate_sampled` must fail if a user has a number of rows other than 101, or if the
item ids of a user differ from the candidate file. That catches a model that scored the
wrong candidates.

Sanity checks before telling the team the evaluation is ready:

| Check | Expected |
|---|---|
| Random, sampled Hit@10 | ≈ 0.099 |
| a model that gives every item score 0 | Hit@10 = 0, NDCG@10 = 0 (all ranks are 101) |
| a model that gives the answer the highest score | Hit@10 = 1, NDCG@10 = 1 |
| PopRec | clearly above Random |

---

## 11. How to write BPR, FMC and FPMC (Kyle)

All three are trained with the same BPR loss: for a user, an item they did interact with
(positive `i`) should score higher than a random item they did not (negative `j`).

```
loss = −ln σ( x(u, i) − x(u, j) )  +  l2 · (squared norms of the vectors used in this example)
```

`σ` is the sigmoid. Use `torch.nn.functional.logsigmoid(x).neg()` instead of
`-torch.log(torch.sigmoid(x))`: the second form becomes `inf` when `x` is very negative.

| Model | Score `x` | Vectors | Training examples from `split.train()` |
|---|---|---|---|
| BPR | `x(u, i) = P[u] · Q[i]` | user table `P`, item table `Q` | every `(u, i)` in user `u`'s training sequence |
| FMC | `x(l, i) = M[l] · N[i]`, `l` = the previous item | two item tables `M`, `N` | every consecutive pair `(l, i)` in a training sequence (the first item has no `l`) |
| FPMC | `x(u, l, i) = P[u] · Q[i] + M[l] · N[i]` | all four tables | same as FMC, plus the user |

Shared settings: Adam, `lr = 0.001`, `batch_size = 1024`, initial weights from a normal
distribution with standard deviation 0.1, one negative per positive, negatives drawn
uniformly from items **not** in the user's training sequence (the same rule SASRec uses).

`score_all` at evaluation time:

| Model | Uses | For validation | For test |
|---|---|---|---|
| BPR | `users` | `P[u] @ Q.T` | the same |
| FMC | last item of `histories[r]` | `l` = last training item | `l` = the validation item (it is the last item of the test history) |
| FPMC | both | both terms | both terms |

Skeleton (fill in the parts marked `TODO`):

```python
@dataclass(frozen=True)
class BPRConfig(ModelConfig):
    d: int = 50
    l2: float = 0.001
    lr: float = 0.001
    batch_size: int = 1024


class BPR(ScoringRecommender):
    name = "bpr"
    config_cls = BPRConfig

    def __init__(self, config: BPRConfig | None = None) -> None:
        self.config: BPRConfig = config or BPRConfig()
        self.user_emb: nn.Embedding | None = None
        self.item_emb: nn.Embedding | None = None

    def fit(self, split: Split) -> None:
        with seeded(self.config.seed):
            # TODO: build user_emb (n_users x d) and item_emb (n_items x d)
            # TODO: list every (u, i) pair from split.train()
            # TODO: each epoch: shuffle, then per batch draw one negative j per pair,
            #       compute the loss above, step the optimizer
            # TODO: after each epoch, validation NDCG@10 + EarlyStopping (Section 8.2)
            ...

    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        if self.user_emb is None or self.item_emb is None:
            raise RuntimeError("call fit() before scoring")
        with torch.no_grad():
            return self.user_emb(torch.tensor(list(users))) @ self.item_emb.weight.T
```

In the paper, BPR loses to PopRec on Beauty (Hit@10 0.3775 vs 0.4003). If yours loses
too, first confirm the grid search ran and the contract tests pass, then report it as a
finding (team plan Phase 1).

---

## 12. EDA (Matthew)

Read the data only through `load_data` and `Split`, so your numbers are the same numbers
the models see.

| Deliverable | How to compute it |
|---|---|
| Table II columns: users, items, avg. actions / user, avg. actions / item, actions | `n_users`, `n_items`, `sum(map(len, split.sequences))` |
| share of users with more than 50 interactions | `len(seq) > 50` over `split.sequences`; give it to Alina (SASRec reads at most 50) |
| per-user and per-item interaction distributions | from `split.sequences`; log scale on both axes |
| users per history group | `history_group(len(h))` for `h` in `split.test()[0]`, and the same for `split.valid()[0]`; give it to Frank |
| duplicates | count of repeated user–item pairs within a sequence; must equal A's number from Section 6.2 |
| time range and order | from the raw `timestamp` column; report the unit you find |

Put figures in `notebooks/eda.ipynb` and save each figure also as a PNG in
`notebooks/figures/` for the report.

---

## 13. Tests every model must pass

Alina provides `tests/conftest.py` with two fixtures:

- `fake_split`: a small random `Split` (60 users, 30 items, 5 to 12 items each);
- `no_test_split`: the same data, but calling `.test()` raises an error.

Add your model to the parametrised contract test. It checks:

```python
@pytest.mark.parametrize("model", [PopRec(), BPR(), ...])
def test_contract(model, no_test_split, fake_split):
    model.fit(no_test_split)  # fails if fit reads test
    histories, answers = fake_split.valid()
    users = list(range(fake_split.n_users))
    assert model.score_all(users, histories).shape == (
        fake_split.n_users,
        fake_split.n_items,
    )
    top = model.recommend(users, histories, 10)
    assert all(len(t) == 10 and not set(t) & set(h) for t, h in zip(top, histories))


def test_same_seed_same_scores(fake_split):
    a, b = BPR(), BPR()
    a.fit(fake_split)
    b.fit(fake_split)
    users, histories = list(range(fake_split.n_users)), fake_split.valid()[0]
    assert torch.equal(a.score_all(users, histories), b.score_all(users, histories))
```

The base class, the metaclass checks of Section 4.2, `Split`, PopRec and the metric
functions of Section 10 were run with pytest before this document was written (9 tests,
all passing, on a small random `Split` with the same checks as above). The BPR skeleton
of Section 11, `seeded`, `EarlyStopping` and `export.py` are interfaces still to be
written.

---

## 14. Workflow

- Python 3.11+, managed with `uv`. Run `uv sync` once after cloning.
- Work on your own branch (`frank/data`, `kyle/baselines`, `matthew/eda`, `alina/sasrec`) and merge
  through a pull request. Pull requests need the CI checks to pass:

  ```bash
  uv run ruff format .
  uv run ruff check .
  uv run mypy src/sasrec2023 tests
  uv run pytest
  ```

- Every public function has a docstring that says what it returns and what it raises.
- Never commit `data/` or `results/*.csv.gz` (they are in `.gitignore`).
- Notebooks: clear the outputs before committing, except the final figure notebooks.

### Order of work

| Step | Who | Unblocks |
|---|---|---|
| 1. `constants.py`, the `Split` class, `config.py`, `registry.py`, `base.py`, `decorators.py`, `conftest.py` | Alina | everyone |
| 2. `load_data`, mapping files, checks of Section 6.2 | Frank | Kyle, Matthew, Alina |
| 3. candidate files, `evaluate.py` lower layer, Random, sanity checks | Frank | Kyle, Alina (early stopping) |
| 4. `export.py`, `run.py`, `EarlyStopping` | Alina | everyone's result files |
| 5. models | Kyle, Alina | Frank (results table) |
| 6. `evaluate.py` upper layer, `results_table.py` | Frank | report |

Steps 1 to 3 are the ones the team plan marks as most likely to block everyone.

---

## 15. Still open

| Question | Who decides | When |
|---|---|---|
| Category (or categories) | Matthew proposes, team decides | Phase 0 meeting |
| History-group limits on the 2023 data | Matthew proposes from the EDA | first half of Phase 1 |
| Shared cloud folder for `results/` | team | Phase 0 meeting |
| What the official files do for the test `history` (Section 6.2) | Frank reports | end of Phase 0 |
