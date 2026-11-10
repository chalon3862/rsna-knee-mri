"""
DICOM -> pre-decoded corpus. Keeps EVERY series of each study (up to 14), tags each with a category
(plane x fluid-sensitive x fat-sat, from DICOM header / SeriesDescription), orders slices geometrically
(ImagePositionPatient projected on the slice normal), crops a fixed physical field of view (130 mm),
normalizes intensities by percentiles, resizes, and flips every knee to the left-knee convention.
Output per shard: vols_<shard>.npy [study, 14 series, 16 slices, H, W], cats, masks, ids.
SHARD is set per copy of the script (Kaggle script kernels take no env vars).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

T0 = time.time()

PRE_SLICES = 16            # halved vs the slot corpus (32) -- series axis grew 6-9 -> 14
PRE_IMG = 336
PRE_BAND = (0.06, 0.94)
PRE_CROP_MM = 130.0
MAX_SERIES = 14            # documented true max series/study (series-study diagnostic, 22-09)
N_SHARD = 4
SHARD = 0                  # <-- set 0/1/2/3 per shard kernel

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
N_CAT = len(CATS) + 1       # + OTHER
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
    """Index into CATS (0..8), or OTHER_CAT if no (plane, fluid, fatsat) combo matches."""
    for k, (_name, plane, fluid, fs) in enumerate(CATS):
        if row["plane"] == plane and bool(row["fatsat"]) == fs and bool(row["fluid"]) == fluid:
            return k
    return OTHER_CAT


def collect_all_series(series_df, plane_map):
    """Every series of every study, tagged with a category id, capped at MAX_SERIES."""
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    series_df["cat"] = series_df.apply(category_of, axis=1)
    out = {}
    n_over_cap = 0
    for study, g in series_df.groupby("StudyInstanceUID"):
        rows = g.sort_values("n_slices", ascending=False)
        if len(rows) > MAX_SERIES:
            n_over_cap += 1
            rows = rows.iloc[:MAX_SERIES]
        out[study] = [r for _, r in rows.iterrows()]
    if n_over_cap:
        log(f"  WARNING: {n_over_cap} studies exceeded MAX_SERIES={MAX_SERIES}, series dropped")
    return out


ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]
DECODE_FAILED = []


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
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
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


def main():
    root = find_root()
    log(f"input root: {root}  |  shard {SHARD}/{N_SHARD}  |  {PRE_SLICES} slices @ {PRE_IMG}px  "
        f"|  MAX_SERIES {MAX_SERIES}  |  {N_CAT} categories (0-{len(CATS) - 1} known, {OTHER_CAT}=OTHER)")

    train_series = pd.read_csv(root / "train_series.csv")
    plane_map = dict(zip(train_series["SeriesInstanceUID"], train_series["Anatomical_Plane"]))

    log("header pass: train")
    htr = annotate(walk("train_series"))
    log(f"  {len(htr)} train series")
    series_map = collect_all_series(htr, plane_map)
    lat_map = lat_of(htr, "train ")

    all_studies = sorted(series_map)
    studies = all_studies[SHARD::N_SHARD]
    log(f"shard {SHARD}: {len(studies)} of {len(all_studies)} studies")
    n_series_this_shard = sum(len(series_map[st]) for st in studies)
    log(f"  {n_series_this_shard} series to decode this shard "
        f"(avg {n_series_this_shard / max(len(studies), 1):.2f}/study)")

    vols = np.lib.format.open_memmap(
        f"vols_{SHARD}.npy", mode="w+", dtype=np.uint8,
        shape=(len(studies), MAX_SERIES, PRE_SLICES, PRE_IMG, PRE_IMG))
    cats = np.full((len(studies), MAX_SERIES), -1, dtype=np.int8)
    mask = np.zeros((len(studies), MAX_SERIES), np.float32)
    log(f"output block: {vols.shape} = {vols.nbytes / 1024 ** 3:.1f} GB (memmap)")

    jobs = [(si, p, row) for si, st in enumerate(studies) for p, row in enumerate(series_map[st])]

    t_ord = time.time()
    log(f"ordering {len(jobs)} series")
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for (_, _, rec), (files, _good) in zip(jobs, pool.map(lambda j: order_slices(j[2]), jobs)):
            rec["ordered"] = files
    log(f"ordered in {time.time() - t_ord:.0f}s")

    t_dec = time.time()
    log(f"decoding {len(jobs)} series")
    done = 0
    CH = 256
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CH):
            block = jobs[c0:c0 + CH]
            for (si, p, rec), img in zip(block, pool.map(
                    lambda j: read_slot_dense(j[2], PRE_SLICES, PRE_IMG), block)):
                done += 1
                if img is None:
                    continue
                plane = rec["plane"]
                vols[si, p] = normalise_laterality(img, plane, lat_map.get(studies[si])).numpy()
                cats[si, p] = int(rec["cat"])
                mask[si, p] = 1.0
            if done % 2048 < CH:
                log(f"  {done}/{len(jobs)}  ({time.time() - t_dec:.0f}s)")
    log(f"decoded in {time.time() - t_dec:.0f}s  |  {int(mask.sum())}/{len(jobs)} series filled"
        f"  |  {len(DECODE_FAILED)} series with a bad slice")

    cat_counts = np.bincount(cats[cats >= 0].ravel(), minlength=N_CAT)
    for k in range(N_CAT):
        name = CATS[k][0] if k < len(CATS) else "OTHER"
        log(f"  category {k} ({name}): {int(cat_counts[k])} series")

    vols.flush()
    np.save(f"cats_{SHARD}.npy", cats)
    np.save(f"masks_{SHARD}.npy", mask)
    np.save(f"ids_{SHARD}.npy", np.array(studies, dtype=object))
    log(f"wrote vols_{SHARD}.npy ({vols.nbytes / 1024 ** 3:.1f} GB) cats_{SHARD}.npy "
        f"masks_{SHARD}.npy ids_{SHARD}.npy")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    log("done")
