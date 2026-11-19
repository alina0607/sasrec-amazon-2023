# data/

The Amazon Reviews 2023 files the notebook reads, one folder per category:

```
data/<Category>/<Category>.train.csv.gz
data/<Category>/<Category>.valid.csv.gz
data/<Category>/<Category>.test.csv.gz
```

Source: the official 5-core `last_out_w_his` files from
https://amazon-reviews-2023.github.io/ (see `docs/team_spec.md`, Section 7.1).
The dataset is released by the McAuley Lab for research use.
