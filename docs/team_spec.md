# How we work on the project notebook

**The whole project is one Jupyter notebook: `sasrec_amazon_2023.ipynb`.** All code goes into
it, written as plainly as possible.

Everything about *what* to build is in the notebook itself:

| Where | What |
|---|---|
| Section 0, "Who does what" | every task, who does it, what it waits for, target date |
| Section 2, "Settings" | the fixed numbers, and a list of where each shared rule is |
| the TODO box at the top of each section | the rules for that part, which the owner builds and presents, and the owner's checklist |

This document only says *how* we work on one notebook together.

---

## 1. What we hand in (due Monday, November 30, 23:55)

| The course asks for | Where it is |
|---|---|
| A notebook with problem definition, data, methods, evaluation, results, discussion; choices explained and justified | `sasrec_amazon_2023.ipynb`, Sections 1 to 10 |
| Clear, well commented code; the notebook runs from top to bottom | all code is in the notebook; CI runs it from top to bottom on every push |
| Data, or a link to it | `data/` in this repository (notebook Section 3.1) |
| Contribution statement agreed by all members | notebook Section 0 |
| A note on which generative AI tools we used and for what | notebook Section 0, every member writes their own row |
| Observations, limitations, future work | notebook Section 10 |
| Demo video | recorded by Matthew from the demo in notebook Section 9; uploaded to Moodle, not committed to git |

The grade is on **depth and how well choices are justified**, not on how many things we try,
and on **code clarity**. Every member must be able to explain their part in their own words
during the Q&A. So:

- prefer a plain function to a clever one;
- every choice gets one sentence of *why* in the Markdown cell above the code;
- if you could not explain a line to the class, rewrite it.

---

## 2. Rules for working on one notebook

A notebook is one file, so two people editing it at the same time can conflict. These rules
keep that rare:

1. **Edit only your own sections.** If you need a change in someone else's section, or in
   a rule that the team agreed on, ask first.
2. **Pull before you start, push soon after you finish.** Small, frequent commits conflict less
   than one large one.
3. **Commit straight to `main`**, no branches or pull requests (it's a small project). Run
   `git pull --rebase` right before you push, and check that CI is green on GitHub afterwards;
   if it's red, fix it first thing.
4. **Outputs are cleared before every commit.** Outputs change on every run and would make a
   conflict in every cell. `pre-commit` clears them for you. The version we hand in is run once
   with outputs, by Matthew, at the end.
5. **Try things in your own draft notebook** in `drafts/` (ignored by git). Move the code into
   the main notebook when it works and passes `check_model`.
6. **Tick off your TODO items** in the notebook as you finish them, so everyone can see where
   the project stands.

---

## 3. Setup on your computer

```bash
git clone https://github.com/alina0607/sasrec-amazon-2023.git
cd sasrec-amazon-2023
uv sync                      # installs everything from uv.lock into .venv
uv run pre-commit install    # once: clears notebook outputs and formats code on commit
uv run jupyter lab           # or open the notebook in VS Code with the .venv kernel
```

Before you push, run the same checks as CI:

```bash
uv run ruff format .
uv run ruff check .
uv run jupyter nbconvert --to notebook --execute sasrec_amazon_2023.ipynb --output-dir /tmp
```

The last line runs the whole notebook from top to bottom. If it fails for you, it fails for the
grader.
