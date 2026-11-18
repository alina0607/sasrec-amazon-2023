"""Numbers that every script of the study shares (docs/team_spec.md, Section 1).

Every result file depends on these values, so change one only after the team
agrees, and never copy a number into another file: import it from here.
"""

# Seed of the 100 negatives in the candidate files.
EVAL_SEED = 0
# Every trained model is run once per seed; tables report mean and std.
TRAIN_SEEDS = (1, 2, 3)
# Negatives ranked against the answer in the sampled evaluation (paper IV-D).
N_NEGATIVES = 100
# Cutoff of Hit@K and NDCG@K (paper IV-D).
K = 10
# Length of the full-ranking result list.
TOP_N = 50
# Epochs without a better validation NDCG@10 before training stops (paper IV-C).
PATIENCE = 20
# Upper bound on epochs, so a run that keeps improving still ends.
MAX_EPOCHS = 200
# Groups of cases by input-history length, as (lowest, highest) with both ends
# included and None for "no upper limit". In 5-core data a test case has at
# least 4 items of history, so a "3 or fewer" group would always be empty.
HISTORY_GROUPS: tuple[tuple[int, int | None], ...] = ((1, 5), (6, 10), (11, None))
