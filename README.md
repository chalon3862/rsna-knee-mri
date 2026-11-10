# RSNA Knee MRI Abnormality Detection — silver medal (134th / 6,425)

Kaggle competition (RSNA 2026): from a multi-series knee MRI (DICOM), predict 12 findings (ACL, MCL, medial/lateral
meniscus, medial/lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture).
Metric: macro ROC-AUC. Code competition: <= 9 h, no internet, T4 GPU.

**Result:** team of two, **134th / 6,425 on the private leaderboard (top 2.1 %) — silver medal**.
This repo contains the part I built alone: the ViT pipeline and the label extraction. My teammate's CNN branch,
his labels and the final blend are not included.

## The problem
4,407 training studies came with **no image annotations**, only multilingual free-text radiology reports.
Only 58 studies were annotated by a radiologist on the images ("gold-58"), used here as validation only.

## Pipeline
```
reports --LLM--> soft labels (labels_llm/)
DICOM --> 1_preprocess.py --> corpus (all series, 16 slices, 130 mm crop, left-knee convention)
corpus --> 2_ssl_pretrain.py --> DINOv2-S continued self-supervised on knee slices
       --> 3_finetune.py     --> 12 findings (spatial attention head)
       --> 4_infer.py        --> submission   (5_eval_gold58.py: validation on gold-58)
several models --> blend_rank.py (rank average)
```
- **Labels**: an LLM reads each report and outputs a probability per finding (rules for ambiguous / unmentioned /
  indirect signs in `labels_llm/PROMPT.md`); ambiguous cases are low, as the gold annotators treat them as negative.
- **Model**: shared DINOv2-S encoder; per study 6 random series x 12 random slices (of 16); head = per-series gated
  patch attention, then per-finding attention over all slices. Weights of epochs 8-10 averaged; inference on all
  series, 9 spatial shifts (TTA).
- **Training**: BCE on soft targets with pos-weights and certainty weights, 2 GPUs (DataParallel on the encoder).

## What moved the leaderboard (public LB, one change at a time)
| Change | Gain |
|---|---|
| Self-supervised pre-training on knee slices | +0.020 |
| LLM labels (instead of a weaker public labelling) | +0.009 |
| Keep all series of each study | +0.006 |
| Spatial attention head | +0.005 |
| 12 of 16 slices at training | +0.005 |
| K=6 series per step | +0.003 |
| Spatial TTA | +0.002 |
| Two models trained on two independent label sets, rank-averaged | +0.002 |

Tried without gain (or worse): 336 px, 2.5D inputs, other backbones (ViT-B, DINOv3), learning-rate variants, SWA,
ordinal severity heads, 105 mm field of view, and **self-distillation** (training on model-corrected targets:
0.952 -> 0.948, the student copies the teacher's biases).

## Lessons
- The public leaderboard is not the truth: my final picks were made on offline metrics fixed in advance, and the
  private ranking moved us **up 105 places** while many teams dropped.
- Data/label quality mattered more than hyper-parameters; label diversity gave the last real gain.
- With a 58-study validation set, differences below ~0.006 are noise: one change at a time, keep a log of dead ends.

## Reproducing
Scripts are Kaggle kernels: paths point to `/kaggle/input/...` and expect the competition data, which is not
redistributed here (see the competition page). Training data, labels derived from the reports, and model weights are
not included. Install: `pip install -r requirements.txt`.
