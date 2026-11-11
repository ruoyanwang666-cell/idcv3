"""Part 2: collect the point-supervision runs (results/ph2_pts_*/metrics.json) into three tables.

  results/weak_summary.md     every run, all test metrics
  results/weak_vs_full.md     point supervision vs full supervision (ph2_unet_bce): gap in Dice
  results/weak_strategies.md  sampling strategies side by side at the same number of clicks

Needs no GPU:  python summarize_weak.py
"""
import json, os
from pathlib import Path
RES = Path(os.environ.get("RESULTS_DIR", "results"))
rows = [json.load(open(p)) for p in sorted(RES.glob("ph2_pts_*/metrics.json"))]
rows.sort(key=lambda r: (r["clicks"]["strategy"], r["clicks"]["pos"] + r["clicks"]["neg"], r["clicks"]["pos"]))
ref = RES / "ph2_unet_bce" / "metrics.json"
ref = json.load(open(ref)) if ref.exists() else None
out = []

# ---------- 1) every run ----------
head = ["| strategy | pos clicks | neg clicks | best epoch | val dice | test dice | test iou | test acc | test sens | test spec | % of full-mask test dice |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
def line(label, p, n, r):
    t = r["test"]
    rel = f"{100 * t['dice'] / ref['test']['dice']:.1f}" if ref else "-"
    return (f"| {label} | {p} | {n} | {r['best_epoch']} | {r['val']['dice']:.4f} | {t['dice']:.4f} | {t['iou']:.4f} | "
            f"{t['acc']:.4f} | {t['sens']:.4f} | {t['spec']:.4f} | {rel} |")
lines = ["## Point supervision on PH2 (U-Net, cross entropy on clicked pixels only)", ""] + head
if ref:
    lines.append(line("full masks (ph2_unet_bce)", "all", "all", ref))
lines += [line(r["clicks"]["strategy"], r["clicks"]["pos"], r["clicks"]["neg"], r) for r in rows]
lines.append("")
(RES / "weak_summary.md").write_text("\n".join(lines)); out += lines

# ---------- 2) point supervision vs full supervision ----------
if ref:
    full = ref["test"]["dice"]
    lines = ["## Point supervision vs full supervision (PH2 test set, U-Net, cross entropy)", "",
             f"Full supervision (all pixels labelled): test Dice {full:.4f}, sensitivity {ref['test']['sens']:.4f}, "
             f"specificity {ref['test']['spec']:.4f}.", "",
             "| strategy | clicks per image (pos + neg) | labelled pixels per image (%) | test dice | difference to full (Dice points) | % of full | test sens | test spec |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, t = r["clicks"], r["test"]
        frac = 100 * (c["pos"] + c["neg"]) / (r["size"] ** 2)
        lines.append(f"| {c['strategy']} | {c['pos']} + {c['neg']} | {frac:.3f} | {t['dice']:.4f} | "
                     f"{100 * (t['dice'] - full):+.1f} | {100 * t['dice'] / full:.1f} | {t['sens']:.4f} | {t['spec']:.4f} |")
    lines.append("")
    for tol in (1, 2, 3, 5):                      # fewest clicks that come within `tol` Dice points of full supervision
        ok = [r for r in rows if 100 * (full - r["test"]["dice"]) <= tol]
        if ok:
            b = min(ok, key=lambda r: r["clicks"]["pos"] + r["clicks"]["neg"])
            lines.append(f"- within {tol} Dice point(s) of full supervision: from {b['clicks']['pos']} + {b['clicks']['neg']} "
                         f"clicks ({b['clicks']['strategy']}, test Dice {b['test']['dice']:.4f})")
        else:
            lines.append(f"- within {tol} Dice point(s) of full supervision: not reached by any run")
    lines.append("")
    (RES / "weak_vs_full.md").write_text("\n".join(lines)); out += lines

# ---------- 3) sampling strategies side by side (balanced runs only) ----------
strategies = sorted({r["clicks"]["strategy"] for r in rows})
bal = {(r["clicks"]["strategy"], r["clicks"]["pos"]): r for r in rows if r["clicks"]["pos"] == r["clicks"]["neg"]}
counts = sorted({n for _, n in bal})
lines = ["## Sampling strategies (PH2 test set, same number of positive and negative clicks)", "",
         "| clicks per class | " + " | ".join(f"{s}: dice | {s}: sens | {s}: spec" for s in strategies) + " |",
         "|---|" + "---|" * (3 * len(strategies))]
for n in counts:
    cells = []
    for s in strategies:
        r = bal.get((s, n))
        cells += [f"{r['test']['dice']:.4f}", f"{r['test']['sens']:.4f}", f"{r['test']['spec']:.4f}"] if r else ["-", "-", "-"]
    lines.append(f"| {n} | " + " | ".join(cells) + " |")
lines.append("")
if len(strategies) < 2:
    lines += ["Only one strategy has been run so far. Run run_strategies.sh to add the random strategy.", ""]
(RES / "weak_strategies.md").write_text("\n".join(lines)); out += lines

print("\n".join(out))
