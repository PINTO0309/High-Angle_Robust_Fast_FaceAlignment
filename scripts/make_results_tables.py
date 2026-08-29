"""結果表の生成(history/050): 各 run の eval_best_{official,stratreal,stress,style}.log を読み、D-ViT(2411.07167 p.6)形式の
markdown 表(NME / FR10・AUC10 / Yaw・Pitch ビン / Roll / カメラ摂動 / 6DRepNet 層別)を標準出力に書く。

使い方: uv run python scripts/make_results_tables.py > tables.md
MODELS の (キー, run dir, 説明) を編集して対象を変える。ログがない表は "-" になる。
"""
import json, re, sys
from pathlib import Path

MODELS = [("vitl-320", "runs/clean_v3", "clean_v3 e198 (ViT-L/16 @320, teacher)"),
          ("vitt-256", "runs/student_s256_96gb_r2", "student_s256_96gb_r2 e449 (ViT-T/16 @256)"),
          ("hg0-256", "runs/student_hg0_wsd", "student_hg0_wsd e386 (PP-HGNetV2-B0 + FPN @256)"),
          ("vitt-096", "runs/student_s096_96gb_r2", "student_s096_96gb_r2 e247 (ViT-T/16 @96, fine-tuned from vitt-256)"),
          ("hg0-096", "runs/student_hg0_s096_wsd", "student_hg0_s096_wsd e134 (PP-HGNetV2-B0 + FPN @96, fine-tuned from hg0-256)")]

def load(run):
    d = Path(run); out = {}
    p = d / "eval_best_official.log"
    if p.exists():
        out["official"] = {m.group(1).strip(): json.loads(m.group(2)) for m in re.finditer(r"\[(.+?)\]\s+(\{.*?\})", p.read_text())}
    p = d / "eval_best_stratreal.log"
    if p.exists():
        t = p.read_text(); out["strat"] = {m.group(1).strip(): (int(m.group(2)), float(m.group(3))) for m in re.finditer(r"^\s+(\|yaw\|\s+[\d.]+|pitch\s+[-\d.]+)\s*:\s*n=\s*(\d+)\s+nme=([\d.]+)", t, re.M)}
        out["strat_mean"] = float(re.search(r"mean_nme=([\d.]+)", t).group(1))
    p = d / "eval_best_stress.log"
    if p.exists():
        t = p.read_text(); st = {"sets": {}, "cfg": {}, "bins": {}}
        for m in re.finditer(r"^\s+(wflw|300w|cofw)\s+base=([\d.]+) worst_roll=([\d.]+) worst_cam=([\d.]+) PS-NME%=([\d.]+) PS-FR@0.1=([\d.]+) (?:degradation|劣化率)=([-+\d.]+)%", t, re.M):
            st["sets"][m.group(1)] = dict(base=float(m.group(2)), worst_roll=float(m.group(3)), worst_cam=float(m.group(4)), ps=float(m.group(5)), psfr=float(m.group(6)), deg=float(m.group(7)))
        for m in re.finditer(r"^\s+(wflw|300w|cofw)\s+(\S+)\s+nme%=([\d.]+) fr=([\d.]+)", t, re.M):
            st["cfg"].setdefault(m.group(1), {})[m.group(2)] = (float(m.group(3)), float(m.group(4)))
        for m in re.finditer(r"^\s+(\|yaw\|\s+[\d.]+|pitch\s+[-\d.]+)\s*:\s*n=\s*(\d+)\s+nme%=([\d.]+)", t, re.M):
            st["bins"][m.group(1).strip()] = (int(m.group(2)), float(m.group(3)))
        out["stress"] = st
    p = d / "eval_best_style.log"
    if p.exists():
        t = p.read_text(); sy = {}
        for m in re.finditer(r"^\s+(wflw_test|300w_valid|cofw_test)\s+(\S+)\s+nme%=([\d.]+)(?: \(([-+\d.]+)%\)| \(n=(\d+)\))", t, re.M):
            sy.setdefault(m.group(1), {})[m.group(2)] = (float(m.group(3)), None if m.group(4) is None else float(m.group(4)))
        out["style"] = sy
    return out

# 併記する外部の公開値(D-ViT, arXiv 2411.07167 v2, p.6 Table 1 / 2 の "Ours" 行。公開ベンチマーク = 未リークの test 値)。
# 本リポジトリの値は全 split 学習の適合度なので同列比較はできない(050 §0)
EXTERNAL = {
    "D-ViT (paper)": {
        "nme": {"WFLW test full": 3.75, "WFLW pose": 6.43, "WFLW expression": 3.85, "WFLW illumination": 4.06, "WFLW makeup": 3.57,
                "WFLW occlusion": 4.47, "WFLW blur": 4.37, "COFW test": 4.13, "300W full": 2.85, "300W common": 2.43, "300W challenge": 4.56},
        "fr": {"WFLW test full": 1.76, "WFLW pose": 8.28, "WFLW expression": 1.27, "WFLW illumination": 1.29, "WFLW makeup": 1.94,
               "WFLW occlusion": 3.80, "WFLW blur": 2.07},
        "auc": {"WFLW test full": 63.7, "WFLW pose": 40.1, "WFLW expression": 62.6, "WFLW illumination": 64.7, "WFLW makeup": 64.7,
                "WFLW occlusion": 57.1, "WFLW blur": 58.6},
    },
}

data = {k: load(run) for k, run, _ in MODELS}
def f(x, nd=2): return "-" if x is None else f"{x:.{nd}f}"
def best_mark(vals, lower=True):
    xs = [v for v in vals if v is not None]
    if not xs: return [None] * len(vals)
    b = min(xs) if lower else max(xs)
    return [v == b for v in vals]

lines = []
# Table 1: NME
keys1 = [("WFLW test full", "Full"), ("WFLW pose", "Pose"), ("WFLW expression", "Expr."), ("WFLW illumination", "Illum."), ("WFLW makeup", "Makeup"), ("WFLW occlusion", "Occl."), ("WFLW blur", "Blur"),
         ("COFW test", "Full"), ("300W full", "Full"), ("300W common", "Comm."), ("300W challenge", "Chal.")]
lines.append("| Model | WFLW Full | Pose | Expr. | Illum. | Makeup | Occl. | Blur | COFW Full | 300W Full | Comm. | Chal. |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, ext in EXTERNAL.items():
    lines.append(f"| {name} | " + " | ".join(f(ext["nme"].get(a)) for a, _ in keys1) + " |")
for k, run, label in MODELS:
    o = data[k].get("official", {})
    lines.append(f"| {k} | " + " | ".join(f(o.get(a, {}).get("nme_pct")) for a, _ in keys1) + " |")
t1 = "\n".join(lines); lines = []
# Table 2: FR10 / AUC10 on WFLW
keys2 = keys1[:7]
lines.append("| Model | FR10 Full | Pose | Exp. | Ill. | Mu. | Occ. | Blur | AUC10 Full | Pose | Exp. | Ill. | Mu. | Occ. | Blur |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, ext in EXTERNAL.items():
    lines.append(f"| {name} | " + " | ".join(f(ext["fr"].get(a)) for a, _ in keys2) + " | " + " | ".join(f(ext["auc"].get(a), 1) for a, _ in keys2) + " |")
for k, run, label in MODELS:
    o = data[k].get("official", {})
    fr = [f(o.get(a, {}).get("fr@0.1", None) * 100 if a in o else None) for a, _ in keys2]
    auc = [f(o.get(a, {}).get("auc@0.1", None) * 100 if a in o else None, 1) for a, _ in keys2]
    lines.append(f"| {k} | " + " | ".join(fr) + " | " + " | ".join(auc) + " |")
t2 = "\n".join(lines); lines = []
# Table 3: yaw / pitch bins (pose-stress effective bins, inter-ocular NME%)
ybins = ["|yaw|    0..30", "|yaw|   30..60", "|yaw|   60..95"]; pbins = ["pitch  -95..-45", "pitch  -45..-15", "pitch  -15..15", "pitch   15..45", "pitch   45..95"]
lines.append("| Model | Yaw 0–30 | Yaw 30–60 | Yaw 60–95 | Pitch −95..−45 | Pitch −45..−15 | Pitch −15..15 | Pitch 15..45 | Pitch 45..95 |")
lines.append("|---|---|---|---|---|---|---|---|---|")
ns = None
for k, run, label in MODELS:
    b = data[k].get("stress", {}).get("bins", {})
    if b and ns is None: ns = [b[x][0] for x in ybins + pbins]
    lines.append(f"| {k} | " + " | ".join(f(b.get(x, (None, None))[1]) for x in ybins + pbins) + " |")
if ns: lines.append("| n | " + " | ".join(str(n) for n in ns) + " |")
t3 = "\n".join(lines); lines = []
# Table 4: roll (pose-stress per-roll rows), per set
rolls = ["base", "roll+045", "roll+090", "roll+135", "roll+180", "roll+225", "roll+270", "roll+315"]
lines.append("| Model | Set | Roll 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 | worst−base |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for k, run, label in MODELS:
    cfg = data[k].get("stress", {}).get("cfg", {})
    for s in ("wflw", "300w", "cofw"):
        c = cfg.get(s, {})
        vals = [c.get(r, (None, None))[0] for r in rolls]
        worst = None if any(v is None for v in vals) else f"{max(vals) - vals[0]:+.2f}"
        lines.append(f"| {k} | {s} | " + " | ".join(f(v) for v in vals) + f" | {worst or '-'} |")
t4 = "\n".join(lines); lines = []
# Table 5: camera pitch / yaw configs
cams = ["cam_p-25", "cam_p-15", "cam_p+15", "cam_p+25", "cam_y-15", "cam_y+15", "cam_p+25_y+15", "cam_p-25_y-15"]
lines.append("| Model | Set | base | cam pitch −25 | −15 | +15 | +25 | cam yaw −15 | +15 | p+25 y+15 | p−25 y−15 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for k, run, label in MODELS:
    cfg = data[k].get("stress", {}).get("cfg", {})
    for s in ("wflw", "300w", "cofw"):
        c = cfg.get(s, {})
        lines.append(f"| {k} | {s} | " + " | ".join(f(c.get(r, (None, None))[0]) for r in ["base"] + cams) + " |")
t5 = "\n".join(lines); lines = []
# Table 6: stratify-real (head-NME x100)
sbins = ["|yaw|    0..30", "|yaw|   30..60", "|yaw|   60..95", "pitch  -90..-30", "pitch  -30..-10", "pitch  -10..10", "pitch   10..30", "pitch   30..90"]
lines.append("| Model | mean | Yaw 0–30 | Yaw 30–60 | Yaw 60–95 | Pitch −90..−30 | Pitch −30..−10 | Pitch −10..10 | Pitch 10..30 | Pitch 30..90 |")
lines.append("|---|---|---|---|---|---|---|---|---|---|")
ns = None
for k, run, label in MODELS:
    b = data[k].get("strat", {})
    if b and ns is None: ns = [b[x][0] for x in sbins]
    lines.append(f"| {k} | {f(data[k].get('strat_mean') * 100 if 'strat_mean' in data[k] else None)} | " + " | ".join(f(b.get(x, (None, None))[1] * 100 if x in b else None) for x in sbins) + " |")
if ns: lines.append("| n | | " + " | ".join(str(n) for n in ns) + " |")
t6 = "\n".join(lines); lines = []
# Table 7: style-shift (NME% and degradation vs clean), rows = model x set
shifts = ["clean", "mblur9", "mblur21", "warm", "cool", "gamma0.6", "gamma1.6", "gray", "jpeg30"]
lines.append("| Model | Set | " + " | ".join(shifts) + " |")
lines.append("|---|---|" + "---|" * len(shifts))
for k, run, label in MODELS:
    sy = data[k].get("style", {})
    for s_ in ("wflw_test", "300w_valid", "cofw_test"):
        c = sy.get(s_, {})
        cells = []
        for sh in shifts:
            v = c.get(sh)
            if v is None: cells.append("-")
            elif v[1] is None: cells.append(f"{v[0]:.2f}")
            else: cells.append(f"{v[0]:.2f} ({v[1]:+.1f}%)")
        lines.append(f"| {k} | {s_} | " + " | ".join(cells) + " |")
t7 = "\n".join(lines)
print("### Table 1 NME (%)\n" + t1 + "\n\n### Table 2 FR10 / AUC10 on WFLW\n" + t2 + "\n\n### Table 3 yaw/pitch bins\n" + t3 + "\n\n### Table 4 roll\n" + t4 + "\n\n### Table 5 camera\n" + t5 + "\n\n### Table 6 stratify-real (head-NME x100)\n" + t6 + "\n\n### Table 7 style-shift (NME%, degradation vs clean)\n" + t7)
