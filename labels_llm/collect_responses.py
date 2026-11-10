"""
Collect the manual Opus 5 batch replies (data/labels/opus5/responses/batch_*.txt)
into the two label tables, and report progress. Safe to run repeatedly as you
add replies over several days.

    python kaggle/nlp/opus_manual_collect.py

Output (rebuilt each run from whatever replies exist):
    data/labels/opus5/llm_states_opus5.csv    uid + 12 state strings
    data/labels/opus5/llm_labels_opus5.csv    uid + 12 soft probabilities
Plus: coverage, unparsed lines, missing uids, and macro-AUC vs the 58 gold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from extract_labels_opus import STATES, TARGETS, state_to_prob

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/labels/opus5"
RESP_DIR = OUT / "responses"
MANIFEST = OUT / "batches" / "_manifest.csv"
TRAIN_CSV = ROOT / "data/raw/train.csv"

_FENCE = re.compile(r"^\s*```(?:json|jsonl)?\s*$|^\s*```\s*$")


def _iter_json_objs(text: str):
    """Yield dicts from a reply: one-per-line JSONL, or a JSON array, tolerant of
    code fences / stray prose lines."""
    stripped = text.strip()
    # whole-reply JSON array
    if stripped.startswith("["):
        try:
            for o in json.loads(stripped):
                if isinstance(o, dict):
                    yield o
            return
        except Exception:
            pass
    for ln in text.splitlines():
        ln = ln.strip().rstrip(",")
        if not ln or _FENCE.match(ln) or not ln.startswith("{"):
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if isinstance(o, dict):
            yield o


def _norm_state(finding: str, state) -> str:
    s = str(state).strip().lower().replace(" ", "_")
    return s if s in STATES[finding] else "not_mentioned"


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST} — run opus_manual_prep.py first")
    man = pd.read_csv(MANIFEST, dtype=str)
    all_uids = list(man["StudyInstanceUID"])
    uid_set = set(all_uids)
    by_batch = man.groupby("batch")["StudyInstanceUID"].apply(list).to_dict()

    states, bad_lines, dup = {}, 0, 0
    replies = sorted(RESP_DIR.glob("batch_*.txt"))
    seen_batches = set()
    for rf in replies:
        tag = rf.stem.split("_")[-1]
        seen_batches.add(tag)
        got = 0
        for o in _iter_json_objs(rf.read_text(encoding="utf-8", errors="ignore")):
            uid = str(o.get("uid") or o.get("StudyInstanceUID") or "").strip()
            if uid not in uid_set:
                bad_lines += 1
                continue
            if uid in states:
                dup += 1
            states[uid] = {t: _norm_state(t, o.get(t, "not_mentioned")) for t in TARGETS}
            got += 1
        exp = len(by_batch.get(tag, []))
        flag = "" if got == exp else f"  <-- expected {exp}"
        print(f"  batch {tag}: {got} parsed{flag}")

    # build tables from whatever we have
    if states:
        sdf = pd.DataFrame(states).T.reindex(columns=TARGETS)
        sdf.index.name = "StudyInstanceUID"
        sdf.reset_index().to_csv(OUT / "llm_states_opus5.csv", index=False)
        pdf = pd.DataFrame({u: {t: state_to_prob(t, s[t]) for t in TARGETS}
                            for u, s in states.items()}).T.reindex(columns=TARGETS)
        pdf.index.name = "StudyInstanceUID"
        pdf.reset_index().to_csv(OUT / "llm_labels_opus5.csv", index=False)

    done = len(states)
    missing_batches = sorted(set(by_batch) - seen_batches)
    missing_uids = [u for u in all_uids if u not in states]
    print(f"\ncoverage: {done}/{len(all_uids)} reports "
          f"({100*done/len(all_uids):.1f}%)  |  {bad_lines} unmatched lines, {dup} dup uids")
    if missing_batches:
        print(f"batches not yet in responses/: {', '.join(missing_batches[:20])}"
              + (" ..." if len(missing_batches) > 20 else ""))
    if 0 < len(missing_uids) <= 40:
        print(f"missing uids: {missing_uids}")

    if done >= 50:
        try:
            from sklearn.metrics import roc_auc_score
            gold = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")[TARGETS].dropna()
            p = pd.read_csv(OUT / "llm_labels_opus5.csv").set_index("StudyInstanceUID")
            ids = p.index.intersection(gold.index)
            if len(ids) >= 20:
                y = gold.loc[ids].values.astype(int)
                pp = p.loc[ids, TARGETS].astype(float).values
                aucs = [roc_auc_score(y[:, j], pp[:, j]) if len(set(y[:, j])) > 1 else np.nan
                        for j in range(12)]
                print(f"\nvs {len(ids)} gold  macro-AUC {np.nanmean(aucs):.4f}   "
                      f"(v4 blend 0.887 | rule 0.796 | wguesdon Opus5 ~0.823)")
                for t, a in zip(TARGETS, aucs):
                    print(f"  {t:18s} {a:.3f}")
        except Exception as e:
            print("gold check skipped:", e)


if __name__ == "__main__":
    main()
