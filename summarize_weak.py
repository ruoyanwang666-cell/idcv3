"""Part 2: collect the point-supervision runs into results/weak_summary.md (test and val metrics),
with the fully supervised U-Net (ph2_unet_bce) as the reference row."""
import json, os
from pathlib import Path
RES = Path(os.environ.get("RESULTS_DIR", "results"))
rows = [json.load(open(p)) for p in sorted(RES.glob("ph2_pts_*/metrics.json"))]
rows.sort(key=lambda r: (r["clicks"]["strategy"], r["clicks"]["pos"] + r["clicks"]["neg"], r["clicks"]["pos"]))
ref = RES / "ph2_unet_bce" / "metrics.json"
ref = json.load(open(ref)) if ref.exists() else None

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
(RES / "weak_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
