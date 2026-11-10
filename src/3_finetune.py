"""
Fine-tuning of the knee-pretrained DINOv2-S on the 12 findings (final recipe).

- Input per study: K=6 series drawn at random among ALL series of the study, 12 slices drawn among 16
  pre-decoded slices per series (true slice positions kept). Inference uses every series and all 16 slices.
- Model: shared DINOv2-S encoder -> all tokens -> spatial head (per-series gated patch attention, then
  per-pathology attention over the series x slices images).
- Targets: soft probabilities from the LLM report reader (labels_llm/), BCE with pos_weight and certainty weights.
- Validation: the 58 gold studies (annotated on the images), never trained on.
- Checkpoint = average of the weights of epochs 8-10.
Inputs: corpus from 1_preprocess.py, backbone from 2_ssl_pretrain.py, LLM labels CSV, competition train.csv.
Output: model_*_{last,bestgold,avgtop3,avglast3}.pth
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
    torch.backends.cudnn.benchmark = True
    log(f"device {dev}  EPOCHS {EPOCHS}  LR_ENC {LR_ENC}  LR_HEAD {LR_HEAD}  N_USE {N_USE}  "
        f"(all-series recipe: K random series/step, joint attention; ALL non-gold studies; gold-58 = validation only)")

    vol_paths, ids, rows, Cats, M = load_pre_index()
    n = len(ids)
    log(f"corpus: {n} studies, MAX_SERIES {MAX_SERIES}, avg present "
        f"{M.sum(1).mean():.2f} series/study (min {int(M.sum(1).min())}, max {int(M.sum(1).max())})")
    train_df = pd.read_csv(find_file("train.csv"))
    P, Wt, gold_idx = load_targets(train_df, ids)
    gold_set = set(gold_idx.tolist())

    log(f"DINOv2 dir: {find_dinov2_dir()}")
    assert "ssl2" in str(find_dinov2_dir()).lower(), "the SSL2 backbone is NOT the one loaded"

    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    probe = RadFT().to(dev)
    log(f"trainable encoder params in the joint phase: {probe.encoder.set_trainable(UNFREEZE_LAST) / 1e6:.1f} M")
    bs, use_ckpt = pick_batch(probe, dev, scaler)
    assert bs == BATCH_STUDIES
    del probe
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    vols = [np.load(p, mmap_mode="r") for p in vol_paths]

    gi = np.array(sorted(gold_set))
    tr = np.array([i for i in range(n) if i not in gold_set])
    gy = (P[gi] > 0.5).astype(int)
    log(f"=== ALL: train {len(tr)} non-gold studies / validation = gold {len(gi)} (never trained on) ===")
    log("LABELS = OPUS v3 (complete); mean label per target: "
        + "  ".join(f"{t[:8]} {v:.3f}" for t, v in zip(TARGETS, P[tr].mean(0))))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = RadFT().to(dev)
    if use_ckpt:
        model.enable_grad_ckpt()
        log("activation checkpointing ON in the ViT (memory) -- same maths, slower")
    log("head (cat_emb + query/out): random init")

    prev = np.clip(P[tr].mean(0), 1e-3, 1 - 1e-3)
    pos_w = torch.from_numpy(np.clip((1.0 - prev) / prev, *POS_WEIGHT_CLIP).astype(np.float32)).to(dev)
    Pt, Wtt = torch.from_numpy(P).to(dev), torch.from_numpy(Wt).to(dev)
    skips = [0]

    def opt_step(loss, opt):
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        s0 = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < s0:
            skips[0] += 1

    def epoch_batches(perm, rng):
        """v2 (I/O fix): ALL random draws of the epoch are made up front, in the SAME order as v1 (pe."""
        out = []
        for b in range(0, len(perm) - bs + 1, bs):
            sel = perm[b:b + bs]
            pos, cat_ids = sample_series(M, Cats, sel, rng, N_USE)
            pick = np.sort(rng.random((len(sel), N_USE, N_SLICE)).argsort(-1)[..., :N_TRAIN], axis=-1)
            out.append((sel, pos, cat_ids, pick))
        return out

    reader = ThreadPoolExecutor(max_workers=READ_THREADS)

    def prefetched(draws):
        futs = {}
        for k in range(min(AHEAD, len(draws))):
            futs[k] = reader.submit(read_pixels, vols, rows, draws[k][0], draws[k][1], draws[k][3])
        for k, (sel, _pos, cat_ids, pick) in enumerate(draws):
            if k + AHEAD < len(draws):
                d = draws[k + AHEAD]
                futs[k + AHEAD] = reader.submit(read_pixels, vols, rows, d[0], d[1], d[3])
            yield sel, cat_ids, pick, futs.pop(k).result()

    def batch_loss(mdl, sel, cat_ids, pick, raw):
        x = augment(resize_gpu(raw, dev))
        cat_t = torch.from_numpy(cat_ids).to(dev)
        mask_t = torch.ones(len(sel), N_USE, device=dev)
        sel_t = torch.as_tensor(sel, device=dev)
        wp = torch.from_numpy(pick.astype(np.float32) / (N_SLICE - 1))    # true pool positions [B,K,N_TRAIN]
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            z = mdl(x, cat_t, mask_t, wp)
        return (F.binary_cross_entropy_with_logits(
            z.float(), Pt[sel_t], reduction="none", pos_weight=pos_w) * Wtt[sel_t]).mean()

    rng = np.random.default_rng(SEED)

    model.encoder.set_trainable(0)
    model.enc_frozen = True
    opt_w = torch.optim.AdamW(model.head.parameters(), lr=LR_HEAD_WARM, weight_decay=WEIGHT_DECAY)
    gstep = 0
    for wep in range(WARM_EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        t_ep, tot, nstep = time.time(), 0.0, 0
        for sel, cat_ids, pick, raw in prefetched(epoch_batches(perm, rng)):
            for g in opt_w.param_groups:
                g["lr"] = LR_HEAD_WARM * min(1.0, (gstep + 1) / 100)
            loss = batch_loss(model, sel, cat_ids, pick, raw)
            opt_step(loss, opt_w)
            gstep += 1
            tot += loss.item()
            nstep += 1
            if nstep % 200 == 0:
                log(f"    warm {wep + 1} step {nstep}/{len(tr) // bs}  loss {tot / nstep:.4f}  "
                    f"skipped {skips[0]}  ({time.time() - t_ep:.0f}s)")
        log(f"  ---- head warm-up {wep + 1}/{WARM_EPOCHS}  loss {tot / max(nstep, 1):.4f}  "
            f"skipped {skips[0]}  ({time.time() - t_ep:.0f}s)")
    del opt_w
    gT0, gA0 = evaluate(model, vols, rows, Cats, M, gi, gy, dev)
    log(f"  after warm-up: mean |g| {model.head.gate.detach().abs().mean().item():.3f}")

    model.enc_frozen = False
    n_tr = model.encoder.set_trainable(UNFREEZE_LAST)
    bb = model.encoder.backbone
    n_blk = len(bb.encoder.layer)
    groups, lrs = [], []
    for i, blk in enumerate(bb.encoder.layer):
        ps = [p for p in blk.parameters() if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": LR_ENC * LLRD ** (n_blk - 1 - i)
                            * (LOW_FACTOR if i < n_blk - HIGH_BLOCKS else 1.0)})
            lrs.append(groups[-1]["lr"])
    groups.append({"params": [p for p in bb.layernorm.parameters() if p.requires_grad], "lr": LR_ENC})
    lrs.append(LR_ENC)
    groups.append({"params": list(model.head.parameters()), "lr": LR_HEAD})
    lrs.append(LR_HEAD)
    log(f"joint phase: {n_tr / 1e6:.1f} M encoder params trainable, LR enc top {LR_ENC} LLRD {LLRD} "
        f"(lowest block LR {min(lrs[:-1]):.2e}) head {LR_HEAD}")
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    steps = EPOCHS * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=steps, pct_start=0.15)

    best_g, best_ep, best_state = -1.0, 0, None
    hist = []
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        t_ep, tot, nstep = time.time(), 0.0, 0
        for sel, cat_ids, pick, raw in prefetched(epoch_batches(perm, rng)):
            loss = batch_loss(model, sel, cat_ids, pick, raw)
            opt_step(loss, opt)
            sched.step()
            tot += loss.item()
            nstep += 1
            if nstep % 200 == 0:
                log(f"    ep {ep + 1} step {nstep}/{len(tr) // bs}  loss {tot / nstep:.4f}  "
                    f"skipped {skips[0]}  ({time.time() - t_ep:.0f}s)")
        log(f"  ---- epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}  skipped {skips[0]}  "
            f"({time.time() - t_ep:.0f}s)")
        gT, gA = evaluate(model, vols, rows, Cats, M, gi, gy, dev)
        hist.append((ep + 1, gT, gA))
        g = model.head.gate.detach().cpu().numpy()
        log("  spatial gate g per pathology: " + "  ".join(f"{t[:8]} {v:+.3f}" for t, v in zip(TARGETS, g))
            + f"   | mean |g| {np.abs(g).mean():.3f}")
        ep_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(ep_state, f"/kaggle/working/ep{ep + 1:02d}_state.pth")
        if gT > best_g:
            best_g, best_ep = gT, ep + 1
            best_state = ep_state
        del ep_state
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def avg_states(epochs):
        acc = None
        for e in epochs:
            st = torch.load(f"/kaggle/working/ep{e:02d}_state.pth", map_location="cpu", weights_only=True)
            if acc is None:
                acc = {k: v.double().clone() for k, v in st.items()}
            else:
                for k in acc:
                    acc[k] += st[k].double()
        return {k: (v / len(epochs)).to(last_state[k].dtype) for k, v in acc.items()}

    top_eps = sorted(e for e, _, _ in sorted(hist, key=lambda h: -h[1])[:TOPK_AVG])
    last_eps = [e for e, _, _ in hist[-TOPK_AVG:]]
    avg_results = {}
    for tag, eps in (("top", top_eps), ("last", last_eps)):
        st = avg_states(eps)
        model.load_state_dict(st)
        gT_avg, gA_avg = evaluate(model, vols, rows, Cats, M, gi, gy, dev)
        avg_results[tag] = (eps, gT_avg, gA_avg, st)
        log(f"  weight-average {tag}{TOPK_AVG} epochs {eps}: gold dense16 {gT_avg:.4f}  4v x 8 {gA_avg:.4f}")

    def save(state, epoch, g, path):
        torch.save({"model": state, "fold": "all", "holdout": float("nan"), "gold": g, "epoch": epoch,
                    "config": {"arch": "dinov2_small_ft_allseries_k6_1216_sh", "img": IMG, "max_series": MAX_SERIES,
                               "n_use_train": N_USE, "n_slice": N_SLICE, "n_cat": N_CAT, "epochs": EPOCHS,
                               "sampling": "K random series/step (train), all present series (eval/infer)",
                               "trained_on": f"all {len(tr)} non-gold studies, gold-58 validation only",
                               "lr_enc": LR_ENC, "lr_head": LR_HEAD, "batch_studies": bs,
                               "warm_epochs": WARM_EPOCHS, "unfreeze_last": UNFREEZE_LAST, "llrd": LLRD,
                               "high_blocks": HIGH_BLOCKS, "low_factor": LOW_FACTOR,
                               "norm": "checkpoint mean/std", "input": "gray_x3_0-1", "feature": "cls+meanpatch + gated per-pathology patch attention (head returns from ALL tokens)",
                               "n_train_slices": N_TRAIN, "grad_ckpt": use_ckpt}},
                   path)

    save(last_state, EPOCHS, hist[-1][1], "/kaggle/working/model_dino_opus_k6_1216_sh_last.pth")
    save(best_state, best_ep, best_g, "/kaggle/working/model_dino_opus_k6_1216_sh_bestgold.pth")
    for tag, (eps, gT_avg, _gA, st) in avg_results.items():
        save(st, eps, gT_avg, f"/kaggle/working/model_dino_opus_k6_1216_sh_avg{tag}{TOPK_AVG}.pth")
    log("---- summary (gold-58; DENSE-16 = main, 4v x 8 = comparison; reference SSL2 all-series K=4, 8 slices, "
        "4-view TTA: warm-up 0.7613, ep8 best 0.9121, ep10 last 0.9095, LB 0.932 / 0.934 +spatial) ----")
    log(f"  warm-up only (head from scratch, encoder frozen): dense16 {gT0:.4f}  4v x 8 {gA0:.4f}")
    for e, gT, gA in hist:
        log(f"  epoch {e}: dense16 {gT:.4f}  4v x 8 {gA:.4f}")
    for tag, (eps, gT_avg, gA_avg, _st) in avg_results.items():
        log(f"  avg{tag}{TOPK_AVG} {eps}: dense16 {gT_avg:.4f}  4v x 8 {gA_avg:.4f}")
    log(f"  saved model_dino_opus_k6_1216_sh_last.pth (epoch {EPOCHS}, gold {hist[-1][1]:.4f}), "
        f"_bestgold.pth (epoch {best_ep}, gold {best_g:.4f}), _avgtop{TOPK_AVG}.pth, _avglast{TOPK_AVG}.pth")


if __name__ == "__main__":
    Path("/kaggle/working").mkdir(exist_ok=True)
    main()
    log("done")
