"""
Inference: weight-averaged checkpoint (epochs 8-10), dense pass over ALL series and all 16 slices of each study,
spatial test-time augmentation (9 small shifts), per-pathology probabilities -> submission.csv.
Runs inside the competition limits (<= 9 h, no internet, T4 GPU). Paths are the Kaggle input paths.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

PRE_SLICES = 16
PRE_IMG = 336
PRE_BAND = (0.06, 0.94)
PRE_CROP_MM = 130.0
N_CAT = 10
N_SLICE = 16                # dense: all 16 pool slices (training drew 12 of them at random)
IMG = 224
TOKEN_DIM = 768

LAT_MIN_OFFSET_MM = 20.0
HDR_THREADS = 16
ORDER_THREADS = 32
PIX_THREADS = 12

CATS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_FLUID_NOFS", "Coronal", True, False),
    ("AX_FLUID_NOFS", "Axial", True, False),
    ("SAG_T1", "Sagittal", False, False),
    ("COR_T1", "Coronal", False, False),
    ("AX_T1", "Axial", False, False),
]
OTHER_CAT = len(CATS)

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("data/raw"), Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
        if (c / "train.csv").is_file():
            return c
    base = Path("/kaggle/input")
    if base.is_dir():
        for d1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [d1] + sorted(p for p in d1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError(f"competition mount not found (cwd {Path.cwd()})")


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


def find_checkpoint():
    for base in (Path("/kaggle/input"), Path(".")):
        if not base.is_dir():
            continue
        for r, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]   # was ~450 s walking the DICOMs
            for f in files:
                if f == "model_dino_opus_k6_1216_sh_avglast3.pth":
                    return Path(r) / f
    raise FileNotFoundError("model_dino_opus_k6_1216_sh_avglast3.pth not attached")


HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept",
            "ImagePositionPatient", "ImageOrientationPatient"]


def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def lat_of(h, tag=""):
    geo = side_from_geometry(h)
    d, n_tag, n_geo, n_none = {}, 0, 0, 0
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f"{tag}laterality: {n_tag} from tag, {n_geo} from geometry, {n_none} unresolved")
    return d


def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series, "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(split):
    base = find_root() / split
    if not base.is_dir():
        return pd.DataFrame(columns=["split", "StudyInstanceUID", "SeriesInstanceUID",
                                     "dir", "files", "n_slices"] + HDR_TAGS)
    items = []
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df


def category_of(row):
    for k, (_name, plane, fluid, fs) in enumerate(CATS):
        if row["plane"] == plane and bool(row["fatsat"]) == fs and bool(row["fluid"]) == fluid:
            return k
    return OTHER_CAT


def collect_all_series(series_df, plane_map):
    """Every series of every study, tagged with a category id."""
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    series_df["cat"] = series_df.apply(category_of, axis=1)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        out[study] = [r for _, r in g.sort_values("n_slices", ascending=False).iterrows()]
    return out


ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]


def order_slices(rec):
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


def normalise_laterality(img, plane, lat):
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


def read_slot_dense(rec, n_slice, out_size):
    files = rec.get("ordered") or rec["files"]
    d, px = rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    lo, hi = int(PRE_BAND[0] * (n - 1)), int(PRE_BAND[1] * (n - 1))
    if hi <= lo:
        lo, hi = 0, n - 1
    idx = np.linspace(lo, hi, n_slice).round().astype(int)
    idx = np.clip(idx, 0, n - 1)

    planes = []
    for i in idx:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)

    got = [k for k, p in enumerate(planes) if p is not None]
    if not got:
        return None
    if len(got) < len(planes):
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    if px and np.isfinite(px) and px > 0:
        want = int(round(PRE_CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


class DinoEncoder(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        from transformers import AutoModel
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

    def forward(self, image):
        """-> ALL tokens [N, 1 + n_patch, D] (CLS first); pooling is done in the head."""
        return self.backbone(pixel_values=(image - self.mean) / self.std).last_hidden_state


class SliceHeadRad(nn.Module):
    """Verbatim from train_dino_ft_k6_1216_sh.py (2-level gated spatial head); inference uses the."""

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

    def forward(self, tok, cat_ids, mask):
        B, S, W, T, D = tok.shape
        tok = tok.float()
        glob = torch.cat([tok[..., 0, :], tok[..., 1:, :].mean(-2)], dim=-1)
        hg = self.proj(glob)
        cid = cat_ids.clamp(min=0)
        hp = self.pproj(tok[..., 1:, :])
        q = self.pquery[None, None] + self.pcat[cid][:, :, None, :]
        sc = torch.einsum("bswph,bsoh->bswop", hp, q) / self.hidden ** 0.5
        pooled = torch.einsum("bswop,bswph->bswoh", sc.softmax(-1), hp)
        h = hg[:, :, :, None, :] + self.gate.view(1, 1, 1, -1, 1) * pooled
        h = h + (self.cat_emb[cid][:, :, None, :] + self.pos_proj(self.winpos))[:, :, :, None, :]
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

    def forward(self, x, cat_ids, mask):
        B, S, W = x.shape[:3]
        z = x.reshape(B * S * W, 1, IMG, IMG).div(255.0).expand(-1, 3, -1, -1)
        f = self.encoder(z)
        f = f.reshape(B, S, W, f.shape[1], f.shape[2])
        return self.head(f, cat_ids, mask)


def resize_gpu(raw_uint8, dev):
    B, K, W = raw_uint8.shape[:3]
    x = torch.from_numpy(raw_uint8).to(dev).float().reshape(-1, 1, PRE_IMG, PRE_IMG)
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return x.reshape(B, K, W, IMG, IMG)


SHIFT_S = 0.03
SHIFTS = [(0.0, 0.0)] + [(dx, dy) for dx in (-SHIFT_S, 0.0, SHIFT_S) for dy in (-SHIFT_S, 0.0, SHIFT_S)
                         if (dx, dy) != (0.0, 0.0)]
SPATIAL_BUDGET_S = 6.0 * 3600
VIEW_BUDGET_S = 7.5 * 3600


def shift_images(x, tx, ty):
    """x [B,S,W,H,W] float 0..255 -> same, translated by (tx, ty) (affine_grid units), border pad."""
    if tx == 0.0 and ty == 0.0:
        return x
    lead = x.shape[:-2]
    y = x.reshape(-1, 1, IMG, IMG)
    theta = torch.zeros(y.shape[0], 2, 3, device=y.device, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 1, 1] = 1.0, 1.0
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    y = F.grid_sample(y, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return y.reshape(*lead, IMG, IMG)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_root()
    log(f"input root: {root}  device {dev}")

    ckpt_path = find_checkpoint()
    log(f"checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert ckpt["config"]["arch"] == "dinov2_small_ft_allseries_k6_1216_sh", ckpt["config"]["arch"]
    log(f"checkpoint config: epoch {ckpt['epoch']}  gold {ckpt['gold']:.4f}")

    model = RadFT().to(dev)
    missing_unexpected = model.load_state_dict(ckpt["model"], strict=True)
    log(f"state_dict load: {missing_unexpected}")
    model.eval()

    test_series = pd.read_csv(root / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))

    log("header pass: test")
    hte = annotate(walk("test_series"))
    log(f"  {len(hte)} test series")
    series_map = collect_all_series(hte, plane_map)
    lat_map = lat_of(hte, "test ")

    studies = sorted(series_map)
    log(f"{len(studies)} test studies")

    sub = {"StudyInstanceUID": studies}
    for t in TARGETS:
        sub[t] = np.full(len(studies), 0.5, np.float32)

    MAX_S_INFER = 20   # generous safety cap (true observed max in train is 14); bounds worst-case
    n_failed = 0
    n_tta = {}            # passes per study -> number of studies (shows if the time guard fired)
    order_pool = ThreadPoolExecutor(max_workers=ORDER_THREADS)
    pix_pool = ThreadPoolExecutor(max_workers=PIX_THREADS)
    for si, st in enumerate(studies):
        recs = series_map[st]
        for rec, (files, _good) in zip(recs, order_pool.map(order_slices, recs)):
            rec["ordered"] = files
        parts = []
        for rec, img in zip(recs, pix_pool.map(
                lambda r: read_slot_dense(r, PRE_SLICES, PRE_IMG), recs)):
            if img is None:
                continue
            img = normalise_laterality(img, rec["plane"], lat_map.get(st))
            parts.append((img.numpy(), int(rec["cat"])))
        if not parts:
            log(f"  WARNING: study {st} has 0 decodable series, left at 0.5")
            continue
        if len(parts) > MAX_S_INFER:
            parts = parts[:MAX_S_INFER]   # collect_all_series already sorted by n_slices desc
        try:
            vols = np.stack([p[0] for p in parts])                 # [S,16,336,336]
            cats = np.array([p[1] for p in parts], dtype=np.int64)  # [S]
            S = vols.shape[0]
            elapsed = time.time() - T0
            shifts = SHIFTS if elapsed < SPATIAL_BUDGET_S else SHIFTS[:1]
            n_tta[len(shifts)] = n_tta.get(len(shifts), 0) + 1
            preds = []
            cat_t = torch.from_numpy(cats)[None].to(dev)
            mask_t = torch.ones(1, S, device=dev)
            x = resize_gpu(vols[None], dev)                          # [1,S,16,IMG,IMG] dense view
            for tx, ty in shifts:
                with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
                    z = model(shift_images(x, tx, ty), cat_t, mask_t).float()
                preds.append(torch.sigmoid(z)[0].cpu().numpy())
            p_mean = np.mean(preds, axis=0)
            for j, t in enumerate(TARGETS):
                sub[t][si] = p_mean[j]
        except Exception as exc:
            n_failed += 1
            log(f"  WARNING: study {st} failed ({exc!r}), left at 0.5")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del parts
        if si % 50 == 0:
            log(f"  {si}/{len(studies)} studies inferred  ({n_failed} failed so far)")
        if si % 100 == 0:
            gc.collect()
    order_pool.shutdown()
    pix_pool.shutdown()
    log(f"done: {n_failed}/{len(studies)} studies failed and were left at 0.5")
    log(f"TTA passes per study -> n studies: {n_tta}  (9 = dense 16 slices x 9 shifts)")

    pd.DataFrame(sub).to_csv("/kaggle/working/submission.csv", index=False)
    log(f"wrote submission.csv ({len(studies)} rows)")


if __name__ == "__main__":
    main()
    log("done")
