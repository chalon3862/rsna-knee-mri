"""Rank-average several submission files (the metric is ROC-AUC, so only the order matters).
Usage: python blend_rank.py out.csv sub_a.csv sub_b.csv [...]   (equal weights)
Each CSV has an id column followed by one probability column per finding."""
import sys
import pandas as pd

out, files = sys.argv[1], sys.argv[2:]
dfs = [pd.read_csv(f) for f in files]
id_col, cols = dfs[0].columns[0], list(dfs[0].columns[1:])
res = dfs[0][[id_col]].copy()
for c in cols:
    res[c] = sum(d[c].rank(pct=True) for d in dfs) / len(dfs)
res.to_csv(out, index=False)
print("wrote", out, res.shape)
