"""
LLM label re-extraction with Claude Opus 5 (Message Batches API).

Two-layer design (per wguesdon's measured result): the model outputs a discrete
STATE per finding from a fixed vocabulary; a deterministic Python layer maps each
state to a soft probability. Asking the model for a calibrated number directly is
asking it to do something it is bad at; asking it to pick from a closed set is not.

The gap the rule extractor leaves is NON-ENGLISH RECALL (rule 0.674 vs LLM 0.833
on non-English reports). So the prompt tells the model to read in the original
language and never translate first.

The competition gold is a radiologist reading the IMAGES, and on-the-fence cases
("possible", "cannot exclude", "mild") are marked NEGATIVE. So a hedge maps to
`equivocal` or lower, never to a confident positive -- this is exactly the
"v4 over-calls positives" failure we measured.

Usage
  pip install anthropic pandas scikit-learn
  export ANTHROPIC_API_KEY=sk-ant-...              # console.anthropic.com key + credits
  python extract_labels_opus.py submit             # creates the batch, prints the id
  python extract_labels_opus.py collect <batch_id> # once it has ended (<= 24h, usually <1h)

Cost (Batch API = 50% off, effort=low): roughly $30-40 for 4407 reports.

On Kaggle instead: a non-competition notebook, internet ON, key in a Kaggle Secret.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data/raw/train.csv"
OUT_DIR = ROOT / "data/labels/opus5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_ID_FILE = OUT_DIR / "batch_id.txt"

MODEL = "claude-opus-5"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

# ---- finding -> allowed states ------------------------------------------------ #
TEAR_STATES = ["complete_tear", "partial_tear", "degeneration_or_high_signal_no_tear",
               "intact_stated", "not_mentioned", "equivocal"]
OA_STATES = ["severe", "moderate", "mild", "absent_stated", "not_mentioned", "equivocal"]
FLUID_STATES = ["large", "moderate", "small_or_trace", "absent_stated", "not_mentioned"]
PRESENCE_STATES = ["present", "absent_stated", "not_mentioned", "equivocal"]
FRACTURE_STATES = ["acute_fracture", "stress_or_insufficiency", "old_or_healed",
                   "absent_stated", "not_mentioned", "equivocal"]

STATES = {
    "ACL": TEAR_STATES, "MCL": TEAR_STATES,
    "Medial Meniscus": TEAR_STATES, "Lateral Meniscus": TEAR_STATES,
    "Medial OA": OA_STATES, "Lateral OA": OA_STATES, "PF OA": OA_STATES,
    "Effusion": FLUID_STATES,
    "Synovitis": PRESENCE_STATES,
    "Baker's": ["present", "absent_stated", "not_mentioned"],
    "Contusion": ["present", "absent_stated", "not_mentioned"],
    "Fracture": FRACTURE_STATES,
}

# ---- deterministic state -> soft probability -------------------------------- #
# not_mentioned uses the per-finding base rate given the report is SILENT. Tunable.
SILENT_BASE = {
    "ACL": 0.10, "MCL": 0.10, "Medial Meniscus": 0.16, "Lateral Meniscus": 0.14,
    "Medial OA": 0.02, "Lateral OA": 0.05, "PF OA": 0.21, "Effusion": 0.15,
    "Synovitis": 0.32, "Baker's": 0.03, "Contusion": 0.08, "Fracture": 0.05,
}
COMMON = {
    "intact_stated": 0.03, "absent_stated": 0.03, "equivocal": 0.45,
}
TEAR_MAP = {"complete_tear": 0.97, "partial_tear": 0.72,
            "degeneration_or_high_signal_no_tear": 0.35, **COMMON}
OA_MAP = {"severe": 0.97, "moderate": 0.90, "mild": 0.62, **COMMON}
FLUID_MAP = {"large": 0.97, "moderate": 0.90, "small_or_trace": 0.55, **COMMON}
PRESENCE_MAP = {"present": 0.93, **COMMON}
FRACTURE_MAP = {"acute_fracture": 0.97, "stress_or_insufficiency": 0.90,
                "old_or_healed": 0.55, **COMMON}
STATE_MAP = {
    "ACL": TEAR_MAP, "MCL": TEAR_MAP,
    "Medial Meniscus": TEAR_MAP, "Lateral Meniscus": TEAR_MAP,
    "Medial OA": OA_MAP, "Lateral OA": OA_MAP, "PF OA": OA_MAP,
    "Effusion": FLUID_MAP, "Synovitis": PRESENCE_MAP,
    "Baker's": PRESENCE_MAP, "Contusion": PRESENCE_MAP, "Fracture": FRACTURE_MAP,
}


def state_to_prob(finding: str, state: str) -> float:
    if state == "not_mentioned":
        return SILENT_BASE[finding]
    return STATE_MAP[finding].get(state, SILENT_BASE[finding])


# ---- prompt ---------------------------------------------------------------- #
SYSTEM = """You extract structured findings from a knee-MRI radiology report.

The report may be in English, Spanish, Turkish, Dutch, German, French, Bulgarian,
Greek, or Serbo-Croatian. Read it in its ORIGINAL language. Do not translate first.

For each of the twelve findings below, output exactly ONE state string from that
finding's allowed set. The states, and what they mean:

TEARS -- ACL, MCL, Medial Meniscus, Lateral Meniscus
  complete_tear            report describes a full-thickness / complete tear or rupture
  partial_tear            report describes a partial tear
  degeneration_or_high_signal_no_tear
                          mucoid/myxoid degeneration, intrasubstance high signal,
                          fraying, or signal that does NOT reach an articular surface
  intact_stated           report explicitly says the structure is intact / normal / continuous
  not_mentioned           the report does not mention this structure at all
  equivocal               "possible", "suspected", "cannot exclude", "questionable"

OSTEOARTHRITIS -- Medial OA, Lateral OA, PF OA  (per compartment)
  severe / moderate / mild
                          cartilage loss, chondral defect/ulcer, chondromalacia grade,
                          joint-space narrowing, or osteophytes, graded for THIS compartment.
                          Whole-joint statements ("tricompartmental OA", "gonarthrosis")
                          apply to all three compartments.
  absent_stated           report says the compartment / cartilage is normal
  not_mentioned / equivocal   as above

EFFUSION
  large / moderate / small_or_trace   graded joint fluid / effusion / epanchement
  absent_stated / not_mentioned       as above

SYNOVITIS
  present                 synovitis, synovial thickening / proliferation / hypertrophy, pannus
  absent_stated / not_mentioned / equivocal

BAKER'S  (popliteal / Baker cyst; a gastrocnemio-semimembranosus bursa counts;
          a meniscal cyst or ganglion does NOT)
  present / absent_stated / not_mentioned

CONTUSION  (bone bruise / bone-marrow oedema / bone contusion)
  present                 explicit bone bruise / contusion / marrow oedema from trauma
  absent_stated / not_mentioned
  NOTE: subchondral marrow oedema UNDER a worn/arthritic compartment is reactive
  degenerative signal, NOT a contusion -- use not_mentioned unless trauma is stated.

FRACTURE
  acute_fracture / stress_or_insufficiency / old_or_healed
  absent_stated / not_mentioned / equivocal
  NOTE: "microfracture" is a surgical procedure, not a fracture.

CRITICAL CALIBRATION. The ground truth is a radiologist reading the IMAGES, and
borderline cases are scored NEGATIVE. When the report hedges -- "possible", "mild",
"cannot exclude", "suspicious for", "questionable", "minimal" -- choose `equivocal`
(or `mild` for OA/effusion), NEVER a confident `complete_tear` / `present`. A
confident state is only for an unambiguous statement.

Output a single JSON object with these twelve keys, values are the state strings:
"ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
"Effusion","Synovitis","Baker's","Contusion","Fracture"
No prose, no explanation, only the JSON object."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": TARGETS,
    "properties": {t: {"type": "string", "enum": STATES[t]} for t in TARGETS},
}


def _client():
    import anthropic
    return anthropic.Anthropic()


def submit():
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    df = pd.read_csv(TRAIN_CSV)[["StudyInstanceUID", "Report"]]
    df["Report"] = df["Report"].fillna("").astype(str)
    reqs = []
    for uid, rep in zip(df["StudyInstanceUID"], df["Report"]):
        reqs.append(Request(
            custom_id=str(uid),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=1200,
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": SCHEMA}},
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user",
                           "content": rep if rep.strip() else "(report empty)"}],
            ),
        ))
    print(f"submitting {len(reqs)} requests as one batch...")
    batch = _client().messages.batches.create(requests=reqs)
    BATCH_ID_FILE.write_text(batch.id)
    print(f"batch id: {batch.id}  (saved to {BATCH_ID_FILE})")
    print(f"status: {batch.processing_status}")
    print(f"collect with:  python {Path(__file__).name} collect {batch.id}")


def collect(batch_id: str):
    client = _client()
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"  {b.processing_status}: {c.processing} processing, {c.succeeded} done, "
              f"{c.errored} errored", flush=True)
        time.sleep(60)
    print(f"ended: {b.request_counts.succeeded} succeeded, {b.request_counts.errored} errored")

    states, probs, bad = {}, {}, []
    for res in client.messages.batches.results(batch_id):
        uid = res.custom_id
        if res.result.type != "succeeded":
            bad.append((uid, res.result.type)); continue
        txt = next((blk.text for blk in res.result.message.content if blk.type == "text"), "")
        try:
            st = json.loads(txt)
        except Exception:
            bad.append((uid, "unparseable")); continue
        st = {t: st.get(t, "not_mentioned") for t in TARGETS}
        states[uid] = st
        probs[uid] = {t: state_to_prob(t, st[t]) for t in TARGETS}

    sdf = pd.DataFrame(states).T.reindex(columns=TARGETS)
    sdf.index.name = "StudyInstanceUID"
    sdf.reset_index().to_csv(OUT_DIR / "llm_states_opus5.csv", index=False)
    pdf = pd.DataFrame(probs).T.reindex(columns=TARGETS)
    pdf.index.name = "StudyInstanceUID"
    pdf.reset_index().to_csv(OUT_DIR / "llm_labels_opus5.csv", index=False)
    print(f"wrote {OUT_DIR/'llm_labels_opus5.csv'} ({len(pdf)} rows); {len(bad)} failed")
    if bad:
        print("  failures:", bad[:10])

    # validate vs the 58 gold
    try:
        from sklearn.metrics import roc_auc_score
        gold = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")[TARGETS].dropna()
        ids = pdf.index.intersection(gold.index)
        y = gold.loc[ids].values.astype(int)
        p = pdf.loc[ids].astype(float).values
        aucs = [roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
                for j in range(12)]
        print(f"\nvs 58 gold  macro-AUC {np.nanmean(aucs):.4f}   "
              f"(v4 blend 0.887, rule 0.796, wguesdon Opus5 ~0.823)")
        for t, a in zip(TARGETS, aucs):
            print(f"  {t:18s} {a:.3f}")
    except Exception as e:
        print("validation skipped:", e)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "submit":
        submit()
    elif cmd == "collect":
        collect(sys.argv[2] if len(sys.argv) > 2 else BATCH_ID_FILE.read_text().strip())
    else:
        print(__doc__)
