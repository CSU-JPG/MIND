"""Extract MIND result scores in the lcm/gsc/visual/dino/action/avg_mse layout.

Schema (discovered by probing existing result JSONs):
  data[i].lcm            -> {avg_mse, avg_psnr, avg_ssim, avg_lpips, length, ...}
  data[i].gsc            -> {avg_gsc?, ...}                # path depends on metric version
  data[i].visual_quality -> {avg_imaging, avg_aesthetic, ...}
  data[i].dino           -> {avg_dino_mse, ...}
  data[i].action         -> {__overall__: {rpe_trans_mean, rpe_rot_mean_deg}, ...}

Each entry in `data` is one staged sample. We mean across samples.
"""
import io
import json
import statistics
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)


def vals(data, metric, key):
    out = []
    for entry in data:
        m = entry.get(metric)
        if isinstance(m, dict):
            v = m.get(key)
            if isinstance(v, (int, float)):
                out.append(v)
    return out


def gsc_vals(data):
    """gsc lives nested at data[i].video_results[0].gsc.mse (a per-frame mse list).
    Per-sample summary = mean of that list. Returns the list of per-sample means."""
    import statistics as _st
    out = []
    for entry in data:
        for vr in entry.get("video_results", []) or []:
            g = vr.get("gsc")
            if isinstance(g, dict):
                mse = g.get("mse")
                if isinstance(mse, list) and mse:
                    out.append(_st.mean(mse))
                break  # first video_results entry only
    return out


def overall(data, metric, key):
    out = []
    for entry in data:
        m = entry.get(metric)
        if isinstance(m, dict) and isinstance(m.get("__overall__"), dict):
            v = m["__overall__"].get(key)
            if isinstance(v, (int, float)):
                out.append(v)
    return out


def first_dict(data, metric):
    """Find any non-empty `metric` dict to probe its keys."""
    for e in data:
        m = e.get(metric)
        if isinstance(m, dict) and m:
            return m
    return None


def find_avg_key(metric_obj, candidates):
    """Pick the first matching avg_* key from a list of candidates."""
    if not metric_obj:
        return None
    for c in candidates:
        if c in metric_obj:
            return c
    return None


def mean(xs):
    return statistics.mean(xs) if xs else None


def fmt(v):
    return "    n/a" if v is None else f"{v:>9.4f}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: _scores_table.py <result_*.json>", file=sys.stderr)
        sys.exit(2)
    d = json.load(open(path, encoding="utf-8"))
    data = d.get("data", [])
    print(f"file:    {Path(path).name}")
    print(f"samples: {len(data)}")
    print()

    # Per-metric: try a list of candidate key names so this works across MIND
    # versions that have evolved key naming.
    lcm_obj = first_dict(data, "lcm")
    gsc_obj = first_dict(data, "gsc")
    vis_obj = first_dict(data, "visual_quality")
    dno_obj = first_dict(data, "dino")

    lcm_psnr_key = find_avg_key(lcm_obj, ["avg_psnr", "psnr_mean", "psnr"])
    lcm_mse_key  = find_avg_key(lcm_obj, ["avg_mse",  "mse_mean",  "mse"])
    gsc_key      = find_avg_key(gsc_obj, ["avg_gsc", "gsc_mean", "gsc"])
    vis_key      = find_avg_key(vis_obj, ["avg_imaging", "avg_musiq", "musiq_mean"])
    dno_key      = find_avg_key(dno_obj, ["avg_dino_mse", "dino_mse_mean", "avg_dino"])

    rows = [
        ("lcm ↑",     vals(data, "lcm",            lcm_psnr_key) if lcm_psnr_key else []),
        ("gsc ↓",     gsc_vals(data)),  # gsc is MSE-based → lower is better
        ("visual ↑",  vals(data, "visual_quality", vis_key)      if vis_key      else []),
        ("dino ↓",    vals(data, "dino",           dno_key)      if dno_key      else []),
        ("action",    overall(data, "action", "rpe_trans_mean")),
        ("avg_mse ↓", vals(data, "lcm",            lcm_mse_key)  if lcm_mse_key  else []),
    ]
    print(f"  {'metric':<12}  {'mean':>9}   n")
    print("  " + "-" * 30)
    for label, xs in rows:
        print(f"  {label:<12}  {fmt(mean(xs))}   {len(xs):>3}")


if __name__ == "__main__":
    main()
