"""
Prepare paste-ready batches for MANUAL Opus 5 label re-extraction via claude.ai
(Claude Pro), for when there is no API budget.

Same two-layer design as extract_labels_opus.py: the model returns a discrete
STATE per finding; opus_manual_collect.py maps states -> soft probabilities.

Each batch file is a SELF-CONTAINED message: the full instructions + N reports,
each tagged with its StudyInstanceUID. Paste one batch file into a fresh (or
reused) claude.ai conversation, copy the model's reply into
data/labels/opus5/responses/batch_<NNN>.txt, move to the next. Order does not
matter; you can stop and resume across days.

    python kaggle/nlp/opus_manual_prep.py [BATCH_SIZE]      # default 50

Output:
    data/labels/opus5/batches/batch_001.txt ... batch_NNN.txt   <- paste these
    data/labels/opus5/batches/_manifest.csv                     <- batch -> uids
    data/labels/opus5/responses/                                <- you fill this
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from extract_labels_opus import SYSTEM, TARGETS   # reuse the crafted prompt + finding list

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data/raw/train.csv"
OUT = ROOT / "data/labels/opus5"
BATCH_DIR = OUT / "batches"
RESP_DIR = OUT / "responses"

BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 50

# One JSON object per report, first key is the uid so a truncated reply is still
# aligned. Keys after uid are exactly TARGETS, values are the state strings.
PER_BATCH_INSTRUCTION = f"""\
You will be given a numbered list of knee-MRI radiology reports, each preceded by
its id on a line starting with `### `.

Apply the rules above to EACH report. Output ONE line of JSON per report, in the
SAME ORDER, nothing else — no code fence, no commentary, no blank lines:

{{"uid": "<the id>", {", ".join(f'"{t}": "<state>"' for t in TARGETS)}}}

If a report is unreadable or empty, still emit its line with every finding set to
"not_mentioned". Do not skip any id. Do not merge lines.
"""


def main():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESP_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)[["StudyInstanceUID", "Report"]]
    df["Report"] = df["Report"].fillna("").astype(str).str.strip()
    df = df.reset_index(drop=True)

    n = len(df)
    n_batch = (n + BATCH_SIZE - 1) // BATCH_SIZE
    manifest = []

    for bi in range(n_batch):
        rows = df.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        tag = f"{bi + 1:03d}"
        body = []
        for uid, rep in zip(rows["StudyInstanceUID"], rows["Report"]):
            body.append(f"### {uid}\n{rep if rep else '(report empty)'}")
            manifest.append({"batch": tag, "StudyInstanceUID": uid})
        text = (
            SYSTEM
            + "\n\n" + "=" * 78 + "\n"
            + PER_BATCH_INSTRUCTION
            + "\n" + "=" * 78 + "\n"
            + f"BATCH {tag} — {len(rows)} reports.\n\n"
            + "\n\n".join(body)
            + "\n"
        )
        (BATCH_DIR / f"batch_{tag}.txt").write_text(text, encoding="utf-8")

    pd.DataFrame(manifest).to_csv(BATCH_DIR / "_manifest.csv", index=False)

    approx_in = int(df["Report"].str.len().mean() / 3.5) * BATCH_SIZE + 1800
    print(f"{n} reports -> {n_batch} batches of {BATCH_SIZE} in {BATCH_DIR}")
    print(f"~{approx_in:,} input tokens/batch (rules + reports), ~{BATCH_SIZE * 120:,} output")
    print(f"manifest: {BATCH_DIR/'_manifest.csv'}")
    print(f"\nworkflow per batch:")
    print(f"  1. open claude.ai, paste the contents of batches/batch_NNN.txt")
    print(f"  2. save the reply to responses/batch_NNN.txt (raw, as-is)")
    print(f"  3. python kaggle/nlp/opus_manual_collect.py   # shows progress + builds the CSVs")


if __name__ == "__main__":
    main()
