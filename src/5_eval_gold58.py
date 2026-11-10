"""
Evaluation on the 58 gold studies (annotated by a radiologist on the images): loads a checkpoint, runs the same
dense inference as 4_infer.py and prints the macro ROC-AUC and the per-finding AUC. The gold set is the only
trustworthy judge here (training labels come from text reports); it is small, so differences below ~0.006
are noise.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

T0 = time.time()

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
PRE_SLICES = 16
PRE_IMG = 336
MAX_SERIES = 14
N_CAT = 10                  # 9 known categories + OTHER
N_USE = 6                   # K=6 series jointly attended per training step (was 4 in the 12/16 run)
N_SLICE = 16                # slices in the pool = what eval/inference sees (dense pass)
N_TRAIN = 12                # RAND-12: slices drawn at random (no replacement) per series per step (was 8, 1 per pair)
IMG = 224
TOKEN_DIM = 768

EPOCHS = 10
WARM_EPOCHS = 1
UNFREEZE_LAST = 12
BATCH_STUDIES = 4
LR_ENC = 3e-5
LLRD = 1.0
HIGH_BLOCKS = 6
LOW_FACTOR = 0.5
TOPK_AVG = 3
READ_THREADS = 4            # v2 I/O fix: background readers
AHEAD = 6                   # v2 I/O fix: batches read ahead of the GPU (same as the SSL scripts)
LR_HEAD_WARM = 1e-3
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.02
SEED = 2026
POS_WEIGHT_CLIP = (1.0, 10.0)
AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, AUG_INTENSITY = 8.0, 0.08, 0.05, 0.10


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_file(name):
    for base in (Path("/kaggle/input"), Path(".")):
        if not base.is_dir():
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            if name in files:
                return Path(r) / name
    raise FileNotFoundError(name)


def _find_pre_parts():
    seen, parts = set(), []

    def _consider(vf):
        key = vf.resolve()
        if key in seen:
            return
        seen.add(key)
        shard = vf.stem.split("_")[-1]
        cf = vf.with_name(f"cats_{shard}.npy")
        mf = vf.with_name(f"masks_{shard}.npy")
        idf = vf.with_name(f"ids_{shard}.npy")
        if cf.is_file() and mf.is_file() and idf.is_file():
            parts.append((shard, vf, cf, mf, idf))

    roots = [b for b in (Path("/kaggle/input"), Path(".")) if b.is_dir()]
    for base in roots:
        for vf in base.glob("*/vols_*.npy"):
            _consider(vf)
    if not parts:
        for base in roots:
            for r, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
                for f in files:
                    if re.fullmatch(r"vols_\w+\.npy", f):
                        _consider(Path(r) / f)
    parts.sort(key=lambda t: t[0])
    return parts


def load_pre_index():
    parts = _find_pre_parts()
    if not parts:
        raise RuntimeError("all-series corpus not attached (kernel_sources rsna-knee-preprocess-allseries-s0..3)")
    vol_paths, ids, rows, cats, masks = [], [], [], [], []
    for si, (shard, vf, cf, mf, idf) in enumerate(parts):
        v = np.load(vf, mmap_mode="r")
        c = np.load(cf)
        m = np.load(mf)
        i = np.load(idf, allow_pickle=True)
        if not (v.shape[0] == c.shape[0] == m.shape[0] == len(i)):
            raise RuntimeError(f"shard {shard} row mismatch")
        if tuple(v.shape[1:]) != (MAX_SERIES, PRE_SLICES, PRE_IMG, PRE_IMG):
            raise RuntimeError(f"shard {shard} vols shape {v.shape} unexpected")
        n = v.shape[0]
        vol_paths.append(str(vf))
        ids.extend(str(x) for x in i)
        rows.append(np.stack([np.full(n, si, np.int32), np.arange(n, dtype=np.int32)], axis=1))
        cats.append(c.astype(np.int64))
        masks.append(m.astype(np.float32))
        log(f"pre-corpus: shard {shard}  {n} studies  {v.nbytes / 1024 ** 3:.1f} GB (mmap)")
        del v
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate StudyInstanceUID across shards")
    return (vol_paths, ids, np.concatenate(rows, axis=0),
            np.concatenate(cats, axis=0), np.concatenate(masks, axis=0))


def sample_series(mask, cats, sel, rng, K):
    """For each study in `sel`, pick K series positions at random among the present ones."""
    B = len(sel)
    pos = np.empty((B, K), dtype=np.int64)
    for b, i in enumerate(sel):
        present = np.where(mask[i] > 0.5)[0]
        pos[b] = rng.choice(present, size=K, replace=len(present) < K)
    cat_ids = cats[np.asarray(sel)[:, None], pos]
    return pos, cat_ids


def read_pixels(vols, rows, sel, pos, pick):
    """pos [B,K] series positions, pick [B,K,N_TRAIN] sorted pool indices -> raw [B,K,N_TRAIN,PRE."""
    B, K = pos.shape
    raw = np.empty((B, K, pick.shape[-1], PRE_IMG, PRE_IMG), np.uint8)
    for b, i in enumerate(sel):
        shard_i, row_i = int(rows[i, 0]), int(rows[i, 1])
        ser = vols[shard_i][row_i, pos[b]]                 # [K, 16, 336, 336]
        raw[b] = ser[np.arange(K)[:, None], pick[b]]
    return raw


def resize_gpu(raw_uint8, dev):
    """raw [B,K,N_SLICE,PRE_IMG,PRE_IMG] uint8 -> [B,K,N_SLICE,IMG,IMG] float 0..255 on dev."""
    B, K, W = raw_uint8.shape[:3]
    x = torch.from_numpy(raw_uint8).to(dev).float().reshape(-1, 1, PRE_IMG, PRE_IMG)
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return x.reshape(B, K, W, IMG, IMG)


def find_dinov2_dir():
    base = Path("/kaggle/input")
    hits = []
    for r, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in r.lower():
            hits.append(Path(r))
    if not hits:
        raise FileNotFoundError("DINOv2 weights not attached")
    for h in hits:
        if "small" in str(h).lower():
            return h
    return hits[0]


class DinoEncoder(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        from transformers import AutoModel
        import json
        path = path or find_dinov2_dir()
        try:
            self.backbone = AutoModel.from_pretrained(str(path), attn_implementation="sdpa")
        except Exception:
            self.backbone = AutoModel.from_pretrained(str(path))
        pc = Path(path) / "preprocessor_config.json"
        cfg = json.loads(pc.read_text()) if pc.is_file() else {}
        mean = cfg.get("image_mean", [0.485, 0.456, 0.406])
        std = cfg.get("image_std", [0.229, 0.224, 0.225])
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.dim = self.backbone.config.hidden_size

    def set_trainable(self, last_n):
        for prm in self.backbone.parameters():
            prm.requires_grad = False
        if last_n > 0:
            layers = self.backbone.encoder.layer
            for blk in layers[max(0, len(layers) - last_n):]:
                for prm in blk.parameters():
                    prm.requires_grad = True
            for prm in self.backbone.layernorm.parameters():
                prm.requires_grad = True
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def forward(self, image):
        """-> ALL tokens [N, 1 + n_patch, D] (CLS first); pooling is done in the head (spatial head)."""
        return self.backbone(pixel_values=(image - self.mean) / self.std).last_hidden_state


class SliceHeadRad(nn.Module):
    """Spatial head (verbatim from train_dino_ft_allseries_spatialhead.py, LB 0.939) + the 12/16 ."""

    def __init__(self, dim, n_cat, n_slice, n_out, hidden=256, p=0.2, tok_dim=384):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.cat_emb = nn.Parameter(torch.randn(n_cat, hidden) * 0.02)
        self.pos_proj = nn.Linear(1, hidden)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden, self.n_slice, self.n_out = hidden, n_slice, n_out
        self.register_buffer("winpos", torch.linspace(0, 1, n_slice).view(1, 1, n_slice, 1))
        self.pproj = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, hidden), nn.GELU())
        self.pquery = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.pcat = nn.Parameter(torch.randn(n_cat, hidden) * 0.02)
        self.gate = nn.Parameter(torch.zeros(n_out))

    def forward(self, tok, cat_ids, mask, winpos=None):
        """tok [B,S,W,1+P,D] all tokens, cat_ids [B,S] long, mask [B,S] (1=present),."""
        B, S, W, T, D = tok.shape
        if winpos is None:
            wp = self.winpos
        elif winpos.dim() == 1:
            wp = winpos.view(1, 1, W, 1).to(tok.device, torch.float32)
        else:
            wp = winpos.view(B, S, W, 1).to(tok.device, torch.float32)
        tok = tok.float()
        glob = torch.cat([tok[..., 0, :], tok[..., 1:, :].mean(-2)], dim=-1)          # [B,S,W,2D]
        hg = self.proj(glob)                                                           # [B,S,W,h]
        cid = cat_ids.clamp(min=0)
        hp = self.pproj(tok[..., 1:, :])                                               # [B,S,W,P,h]
        q = self.pquery[None, None] + self.pcat[cid][:, :, None, :]                    # [B,S,O,h]
        sc = torch.einsum("bswph,bsoh->bswop", hp, q) / self.hidden ** 0.5
        pooled = torch.einsum("bswop,bswph->bswoh", sc.softmax(-1), hp)                # [B,S,W,O,h]
        h = hg[:, :, :, None, :] + self.gate.view(1, 1, 1, -1, 1) * pooled
        h = h + (self.cat_emb[cid][:, :, None, :] + self.pos_proj(wp))[:, :, :, None, :]
        h = h.reshape(B, S * W, self.n_out, self.hidden)
        att = torch.einsum("btoh,oh->bot", h, self.query) / self.hidden ** 0.5
        m = mask[:, :, None].expand(B, S, W).reshape(B, S * W)
        att = att.masked_fill(m.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bot,btoh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class RadFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        assert self.encoder.dim * 2 == TOKEN_DIM, self.encoder.dim
        self.head = SliceHeadRad(TOKEN_DIM, N_CAT, N_SLICE, len(TARGETS), tok_dim=self.encoder.dim)
        self.enc_frozen = False

    def enable_grad_ckpt(self):
        """Activation checkpointing in the ViT blocks: identical gradients, less memory, ~+30 % compu."""
        self.encoder.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, x, cat_ids, mask, winpos=None):
        """x: [B,S,W,H,W] float 0..255 (already augmented)."""
        B, S, W = x.shape[:3]
        z = x.reshape(B * S * W, 1, IMG, IMG).div(255.0).expand(-1, 3, -1, -1)
        if self.enc_frozen:
            with torch.no_grad():
                f = self.encoder(z)
        else:
            f = self.encoder(z)
        return self.head(f.reshape(B, S, W, f.shape[1], f.shape[2]), cat_ids, mask, winpos)


def augment(x):
    lead = x.shape[:-2]
    y = x.reshape(-1, 1, IMG, IMG)
    n, dev = y.shape[0], y.device
    rot = (torch.rand(n, device=dev) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev) * AUG_SCALE
    tx = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 2 * AUG_INTENSITY
    y = (y * scale).clamp(0, 255)
    return y.reshape(*lead, IMG, IMG)


def auc_or_nan(y, p):
    return roc_auc_score(y, p) if len(set(y)) > 1 else np.nan


def macro_auc(y, p):
    return float(np.nanmean([auc_or_nan(y[:, j], p[:, j]) for j in range(y.shape[1])]))


@torch.no_grad()
def evaluate(model, vols, rows, cats, mask, gi, gy, dev):
    """Joint attention over ALL available series of each gold study (variable S)."""
    model.eval()
    n8 = 8
    picks = {
        "A": np.zeros(n8, np.int64), "B": np.ones(n8, np.int64),
        "C": (np.arange(n8) % 2).astype(np.int64), "D": (1 - np.arange(n8) % 2).astype(np.int64),
    }
    dense, per_view = [], {v: [] for v in picks}
    for i in gi:
        present = np.where(mask[i] > 0.5)[0]
        shard_i, row_i = int(rows[i, 0]), int(rows[i, 1])
        cat_ids = torch.from_numpy(cats[i, present]).to(dev)[None]
        mask_t = torch.ones(1, len(present), device=dev)
        ser = vols[shard_i][row_i, present]                       # read the study's series ONCE
        x = resize_gpu(np.ascontiguousarray(ser)[None], dev)      # [1, S_i, 16, IMG, IMG]
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            z = model(x, cat_ids, mask_t).float()
        dense.append(torch.sigmoid(z)[0].cpu().numpy())
        for v, pk in picks.items():
            idx = 2 * np.arange(n8) + pk
            raw = ser[:, idx]                                     # [S_i, 8, 336, 336]
            x = resize_gpu(raw[None], dev)
            wp = torch.from_numpy(idx.astype(np.float32) / (N_SLICE - 1))
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(x, cat_ids, mask_t, wp).float()
            per_view[v].append(torch.sigmoid(z)[0].cpu().numpy())
    evaluate.pred = np.stack(dense)
    evaluate.pred4 = np.mean([np.stack(p) for p in per_view.values()], axis=0)
    gD = macro_auc(gy, np.stack(dense))
    gV = macro_auc(gy, np.mean([np.stack(p) for p in per_view.values()], axis=0))
    log(f"  gold(n={len(gi)})  DENSE-16 (main) {gD:.4f}   4 views x 8 (comparison) {gV:.4f}")
    return gD, gV


def pick_batch(model, dev, scaler):
    """Batch is FIXED at BATCH_STUDIES (no silent change of recipe)."""
    if dev.type != "cuda":
        return BATCH_STUDIES, False
    model.train()
    bs = BATCH_STUDIES
    for use_ckpt in (False, True):
        if use_ckpt:
            model.enable_grad_ckpt()
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, 256, (bs, N_USE, N_TRAIN, IMG, IMG), dtype=torch.uint8, device=dev).float()
            cat_t = torch.randint(0, N_CAT, (bs, N_USE), device=dev)
            m = torch.ones(bs, N_USE, device=dev)
            wp = torch.linspace(0, 1, N_TRAIN)
            with torch.autocast("cuda"):
                loss = model(augment(x), cat_t, m, wp).float().mean()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            log(f"batch check OK: {bs} studies/step ({bs * N_USE * N_TRAIN} imgs), grad ckpt {use_ckpt}, "
                f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB")
            model.zero_grad(set_to_none=True)
            del x, cat_t, m, loss
            torch.cuda.empty_cache()
            return bs, use_ckpt
        except torch.cuda.OutOfMemoryError:
            log(f"OOM at {bs} studies/step (grad ckpt {use_ckpt})")
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"{bs} studies/step does not fit even with activation checkpointing")


def load_targets(train_df, ids):
    lab = pd.read_csv(find_file("llm_labels_opus5_v3.csv"))
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    lab = lab.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(ids)
    P = lab[TARGETS].values.astype(np.float32)
    assert not np.isnan(P).any(), "Opus v3 labels missing for some studies (partial file mounted?)"
    assert P.min() >= 0.0 and P.max() <= 1.0
    W = (0.25 + 0.75 * (2.0 * np.abs(P - 0.5))).astype(np.float32)
    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    gi = np.array([i for i, s in enumerate(ids) if s in gold.index])
    P[gi] = np.array([gold.loc[ids[i]].values for i in gi]).astype(np.float32)
    W[gi] = 3.0
    return P, W, gi


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_paths, ids, rows, Cats, M = load_pre_index()
    train_df = pd.read_csv(find_file("train.csv"))
    P, _W, gi = load_targets(train_df, ids)
    gi = np.array(sorted(gi.tolist()))
    gy = (P[gi] > 0.5).astype(int)
    lab = pd.read_csv(find_file("llm_labels_opus5_v3.csv"))
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    O = lab.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").loc[[ids[i] for i in gi], TARGETS].values
    model = RadFT().to(dev)
    ck = torch.load(find_file("model_dino_opus_k6_1216_sh_avglast3.pth"), map_location="cpu", weights_only=True)
    res = model.load_state_dict(ck["model"])
    log(f"checkpoint: epoch {ck['epoch']} stored gold {ck['gold']:.4f} | load: {res}")
    vols = [np.load(p, mmap_mode="r") for p in vol_paths]
    gD, gV = evaluate(model, vols, rows, Cats, M, gi, gy, dev)
    D, V = evaluate.pred, evaluate.pred4
    log(f"gold macro dense16 {gD:.4f} (stored {ck['gold']:.4f})  4v x 8 {gV:.4f}  Opus labels {macro_auc(gy, O):.4f}")
    log(f"{'pathology':17s} pos  model-dense16  model-4v  Opus-label  model-Opus")
    for j, t in enumerate(TARGETS):
        a_d, a_v, a_o = auc_or_nan(gy[:, j], D[:, j]), auc_or_nan(gy[:, j], V[:, j]), auc_or_nan(gy[:, j], O[:, j])
        log(f"{t:17s} {gy[:, j].sum():3d}  {a_d:12.3f}  {a_v:8.3f}  {a_o:10.3f}  {a_d - a_o:+10.3f}")
    gid = np.array([ids[i] for i in gi])
    np.savez_compressed("/kaggle/working/gold_preds_952.npz", gold_ids=gid, gold_dense=D, gold_4v=V, gold_y=gy,
                        opus=O, targets=np.array(TARGETS))
    pd.DataFrame(D, columns=TARGETS).assign(StudyInstanceUID=gid).to_csv("/kaggle/working/gold_preds_952.csv", index=False)
    log("saved gold_preds_952.npz / .csv")


if __name__ == "__main__":
    Path("/kaggle/working").mkdir(exist_ok=True)
    main()
