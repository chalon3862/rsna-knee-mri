"""
Self-supervised continuation of DINOv2-S on the knee-MRI slices of the training set (no labels, DINO-style
self-distillation: student sees 2 global 224 px + 6 local 98 px crops, EMA teacher sees the global ones,
centered + sharpened teacher outputs avoid collapse). No flips (laterality is a signal).
Backbone LR is 0 for the first steps (the new head matches the teacher first), then warm-up + cosine.
Data: corpus from 1_preprocess.py, one fold held out. Output: HF-format teacher backbone.
This domain pre-training gave +0.020 on the public leaderboard.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import copy
import hashlib
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

PRE_SLICES = 32
PRE_IMG = 336
N_SLOT_CORPUS = 6
NFOLD = 5
HOLDOUT_FOLD = 4                      # excluded from the SSL data (clean later comparison)

BATCH = int(os.environ.get("BATCH", 32))            # base images per step
N_LOCAL = 6
G_IMG, L_IMG = 224, 98                # 16x16 and 7x7 patches of 14
G_SCALE, L_SCALE = (0.40, 1.00), (0.10, 0.40)
STEPS = int(os.environ.get("STEPS", 13000))
K_PROTO, HID, BOT = 4096, 2048, 256
LR_BB = float(os.environ.get("LR_BB", 1e-5))
LR_HEAD = float(os.environ.get("LR_HEAD", 5e-4))
WD = 0.04
WARM_FRAC = 0.05
FREEZE_BB_STEPS = int(os.environ.get("FREEZE_BB_STEPS", 300))
TAU_S, TAU_T, CENTER_M = 0.1, 0.04, 0.9
EMA_M0 = 0.996
CLIP = 3.0
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_MIN", 170)) * 60
SAVE_EVERY = 2000
SEED = 2026
OUT_DIR = Path("/kaggle/working/dinov2_knee_ssl")


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


def find_dinov2_dir():
    hits = []
    for r, dirs, files in os.walk(Path("/kaggle/input")):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in r.lower():
            hits.append(Path(r))
    if not hits:
        raise FileNotFoundError("DINOv2 weights not attached (model_sources metaresearch/dinov2)")
    for h in hits:
        if "small" in str(h).lower():
            return h
    return hits[0]


def _find_pre_parts():
    seen, parts = set(), []

    def _consider(vf):
        key = vf.resolve()
        if key in seen:
            return
        seen.add(key)
        shard = vf.stem.split("_")[-1]
        mf, idf = vf.with_name(f"masks_{shard}.npy"), vf.with_name(f"ids_{shard}.npy")
        if mf.is_file() and idf.is_file():
            parts.append((shard, vf, mf, idf))

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


def build_candidates():
    """-> vols (list of memmaps), cand int32 [M,3] = (shard, row, slot) of every filled slot of e."""
    parts = _find_pre_parts()
    if not parts:
        raise RuntimeError("pre-decoded corpus not attached (kernel_sources rsna-knee-preprocess-s0..3)")
    train_df = pd.read_csv(find_file("train.csv"))
    rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    vols, cand, n_all, n_held = [], [], 0, 0
    for si, (shard, vf, mf, idf) in enumerate(parts):
        v = np.load(vf, mmap_mode="r")
        m = np.load(mf)
        ids = [str(x) for x in np.load(idf, allow_pickle=True)]
        if not (v.shape[0] == m.shape[0] == len(ids)) or tuple(v.shape[1:]) != (N_SLOT_CORPUS, PRE_SLICES, PRE_IMG, PRE_IMG):
            raise RuntimeError(f"shard {shard}: unexpected shape {v.shape}")
        fold = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % NFOLD for s in ids])
        keep_study = fold != HOLDOUT_FOLD
        n_all += len(ids)
        n_held += int((~keep_study).sum())
        valid = (m > 0.5) & keep_study[:, None]
        rc = np.argwhere(valid)                                   # [(row, slot)]
        cand.append(np.concatenate([np.full((len(rc), 1), si, np.int32), rc.astype(np.int32)], axis=1))
        vols.append(v)
        log(f"shard {shard}: {len(ids)} studies, {int(valid.sum())} usable slots  ({v.nbytes / 1024 ** 3:.1f} GB mmap)")
    cand = np.concatenate(cand, axis=0)
    log(f"studies {n_all} (holdout fold {HOLDOUT_FOLD} excluded: {n_held}); candidate slots {len(cand)}; "
        f"candidate slices {len(cand) * PRE_SLICES}")
    return vols, cand


def read_batch(vols, cand, step):
    rng = np.random.default_rng([SEED, step])
    pick = rng.integers(0, len(cand), BATCH)
    sl = rng.integers(0, PRE_SLICES, BATCH)
    out = np.empty((BATCH, PRE_IMG, PRE_IMG), np.uint8)
    for j in range(BATCH):
        sh, row, slot = cand[pick[j]]
        out[j] = vols[sh][row, slot, sl[j]]
    return out


def crops(x, n, out, scale):
    """x [B,1,H,W] float 0..1 -> [n*B,1,out,out] random resized crops (n per image), bilinear, bo."""
    dev = x.device
    xr = x.repeat(n, 1, 1, 1)
    N = xr.shape[0]
    area = torch.empty(N, device=dev).uniform_(*scale)
    r = torch.exp(torch.empty(N, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)))
    w = torch.sqrt(area * r).clamp(max=1.0)
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(N, device=dev) * 2 - 1) * (1 - w)
    cy = (torch.rand(N, device=dev) * 2 - 1) * (1 - h)
    ang = (torch.rand(N, device=dev) * 2 - 1) * (8.0 * math.pi / 180)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(N, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = w * c, -h * s, cx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = w * s, h * c, cy
    grid = F.affine_grid(theta, (N, 1, out, out), align_corners=False)
    return F.grid_sample(xr, grid, mode="bilinear", padding_mode="border", align_corners=False)


def photometric(y):
    N, dev = y.shape[0], y.device
    gamma = torch.exp(torch.empty(N, 1, 1, 1, device=dev).uniform_(-0.3, 0.3))
    y = y.clamp(1e-4, 1.0) ** gamma
    y = (y - 0.5) * torch.empty(N, 1, 1, 1, device=dev).uniform_(0.8, 1.2) + 0.5 \
        + torch.empty(N, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
    y = y + torch.randn_like(y) * torch.empty(N, 1, 1, 1, device=dev).uniform_(0.0, 0.03)
    return y.clamp(0.0, 1.0)


class DinoHead(nn.Module):
    def __init__(self, dim, hid, bot, k):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, bot))
        self.proto = nn.Parameter(torch.randn(k, bot) * 0.02)

    def forward(self, x):
        z = F.normalize(self.mlp(x), dim=-1)
        return z @ F.normalize(self.proto, dim=-1).T               # cosine logits in [-1, 1]


class Net(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import AutoModel
        try:
            self.backbone = AutoModel.from_pretrained(str(path), attn_implementation="sdpa")
        except Exception:
            self.backbone = AutoModel.from_pretrained(str(path))
        self.head = DinoHead(self.backbone.config.hidden_size, HID, BOT, K_PROTO)

    def forward(self, x):
        return self.head(self.backbone(pixel_values=x).last_hidden_state[:, 0])


def make_groups(net):
    groups = {"bb_decay": [], "bb_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in net.named_parameters():
        kind = "bb" if name.startswith("backbone.") else "head"
        groups[f"{kind}_{'nodecay' if p.ndim <= 1 else 'decay'}"].append(p)
    return [{"params": ps, "kind": k, "weight_decay": (WD if k.endswith("_decay") else 0.0)} for k, ps in groups.items()]


def save_backbone(teacher, src_dir, tag):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher.backbone.save_pretrained(str(OUT_DIR))
    pc = Path(src_dir) / "preprocessor_config.json"
    if pc.is_file():
        shutil.copy(pc, OUT_DIR / "preprocessor_config.json")
    log(f"saved teacher backbone ({tag}) -> {OUT_DIR}")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert dev.type == "cuda", "GPU required"
    torch.manual_seed(SEED)
    log(f"device {dev}  BATCH {BATCH} (x {2 + N_LOCAL} views)  STEPS {STEPS}  LR_BB {LR_BB}  LR_HEAD {LR_HEAD}  "
        f"freeze backbone {FREEZE_BB_STEPS} steps  budget {TIME_BUDGET_S / 60:.0f} min")
    vols, cand = build_candidates()

    src = find_dinov2_dir()
    log(f"DINOv2 dir: {src}")
    cfg = json.loads((src / "preprocessor_config.json").read_text()) if (src / "preprocessor_config.json").is_file() else {}
    mean = torch.tensor(cfg.get("image_mean", [0.485, 0.456, 0.406]), device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("image_std", [0.229, 0.224, 0.225]), device=dev).view(1, 3, 1, 1)

    student = Net(src).to(dev)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    student.train()
    log(f"backbone params {sum(p.numel() for p in student.backbone.parameters()) / 1e6:.1f} M, "
        f"head params {sum(p.numel() for p in student.head.parameters()) / 1e6:.1f} M")

    opt = torch.optim.AdamW(make_groups(student), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda")
    center = torch.zeros(1, K_PROTO, device=dev)
    sp = [p for p in student.parameters()]
    tp = [p for p in teacher.parameters()]

    def norm(y):
        return (y.expand(-1, 3, -1, -1) - mean) / std

    ex = ThreadPoolExecutor(max_workers=4)
    AHEAD = 6
    futs = {s: ex.submit(read_batch, vols, cand, s) for s in range(min(AHEAD, STEPS))}
    hist, run_loss, t_last, n_done = [], 0.0, time.time(), 0
    for step in range(STEPS):
        if time.time() - T0 > TIME_BUDGET_S:
            log(f"time budget reached at step {step}/{STEPS}: stopping")
            break
        warm = min(1.0, (step + 1) / (WARM_FRAC * STEPS))
        f = warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            base = LR_BB if g["kind"].startswith("bb") else LR_HEAD
            g["lr"] = 0.0 if (g["kind"].startswith("bb") and step < FREEZE_BB_STEPS) else base * f
        if step + AHEAD < STEPS:
            futs[step + AHEAD] = ex.submit(read_batch, vols, cand, step + AHEAD)
        raw = torch.from_numpy(futs.pop(step).result()).to(dev, non_blocking=True)
        x = raw.float().div_(255.0).unsqueeze(1)                          # [B,1,336,336] in 0..1

        with torch.no_grad():
            g_views = torch.cat([photometric(crops(x, 1, G_IMG, G_SCALE)) for _ in range(2)], 0)     # [2B,1,224,224]
            l_views = photometric(crops(x, N_LOCAL, L_IMG, L_SCALE))                                  # [6B,1,98,98]
            g_in, l_in = norm(g_views), norm(l_views)

        with torch.no_grad(), torch.autocast("cuda"):
            t_logits = teacher(g_in).float()                                                            # [2B,K]
        t_prob = F.softmax((t_logits - center) / TAU_T, dim=-1)
        t_chunks = t_prob.chunk(2)

        with torch.autocast("cuda"):
            s_g = student(g_in).float()
            s_l = student(l_in).float()
        s_logp = [F.log_softmax(v / TAU_S, dim=-1) for v in list(s_g.chunk(2)) + list(s_l.chunk(N_LOCAL))]
        loss, npair = 0.0, 0
        for i, tc in enumerate(t_chunks):
            for j, sl_ in enumerate(s_logp):
                if j == i:
                    continue
                loss = loss + torch.sum(-tc * sl_, dim=-1).mean()
                npair += 1
        loss = loss / npair
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(sp, CLIP)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            m = 1.0 - (1.0 - EMA_M0) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
            torch._foreach_mul_(tp, m)
            torch._foreach_add_(tp, sp, alpha=1.0 - m)
            center.mul_(CENTER_M).add_(t_logits.mean(0, keepdim=True), alpha=1.0 - CENTER_M)

        run_loss += float(loss)
        n_done += 1
        if step in (0, 1, 2, 9, 49) or (step + 1) % 100 == 0:
            with torch.no_grad():
                ent = -(t_prob * torch.log(t_prob + 1e-12)).sum(-1).mean().item()
                pm = t_prob.mean(0)
                bat_ent = -(pm * torch.log(pm + 1e-12)).sum().item()
            dt = time.time() - t_last
            rate = n_done / max(dt, 1e-6)
            hist.append({"step": step + 1, "loss": run_loss / n_done, "t_ent": ent, "bat_ent": bat_ent})
            log(f"  step {step + 1}/{STEPS}  loss {run_loss / n_done:.4f}  teacher entropy {ent:.3f} "
                f"(max {math.log(K_PROTO):.2f})  batch-mean entropy {bat_ent:.3f}  lr_bb {opt.param_groups[0]['lr']:.2e}  "
                f"ema {m:.4f}  {rate:.2f} step/s  peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB")
            run_loss, t_last, n_done = 0.0, time.time(), 0
        if (step + 1) % SAVE_EVERY == 0 and step + 1 < STEPS:
            save_backbone(teacher, src, f"step {step + 1}")

    save_backbone(teacher, src, "final")
    Path("/kaggle/working/ssl_log.json").write_text(json.dumps(hist), encoding="utf-8")
    log("---- done. collapse check: teacher entropy should stay well below log(K) but batch-mean entropy should stay high "
        "(not near 0). Next: supervised warm-up with this backbone frozen (gate: holdout 0.7306 with the original DINOv2).")


if __name__ == "__main__":
    Path("/kaggle/working").mkdir(exist_ok=True)
    main()
