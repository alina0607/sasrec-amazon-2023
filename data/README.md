# data/

The notebook downloads the Amazon Reviews 2023 files here on its first run (Section 3.1), one
folder per category:

```
data/<Category>/<Category>.train.csv.gz
data/<Category>/<Category>.valid.csv.gz
data/<Category>/<Category>.test.csv.gz
```

We use `Industrial_and_Scientific` (about 14 MB in total). The files are not committed to git:
every run gets the same official files from the source.

Source: the official 5-core `last_out_w_his` files from
https://amazon-reviews-2023.github.io/data_processing/5core.html.
The dataset is released by the McAuley Lab for research use.
