"""Collect results/*/metrics.json into results/summary.md (test and val metrics)."""
import json
from pathlib import Path
import os
RES = Path(os.environ.get("RESULTS_DIR", "results"))
rows = [json.load(open(p)) for p in sorted(RES.glob("*/metrics.json"))]
lines = []
for split in ("test", "val", "train"):
    lines += [f"### {split}", "", "| experiment | dataset | model | loss | best epoch | dice | iou | acc | sens | spec |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r[split]
        lines.append(f"| {r['name']} | {r.get('dataset','drive')} | {r['model']} | {r['loss']} | {r['best_epoch']} | {m['dice']:.4f} | {m['iou']:.4f} | "
                     f"{m['acc']:.4f} | {m['sens']:.4f} | {m['spec']:.4f} |")
    lines.append("")
(RES / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))

# ---- loss ablation table (U-Net): bce vs focal vs wbce, per dataset ----
by = {r["name"]: r for r in rows}
abl = ["## Loss-function ablation (U-Net, only the loss changes)", ""]
for ds in ("drive", "ph2"):
    found = [(l, by.get(f"{ds}_unet" if l == "wbce" else f"{ds}_unet_{l}")) for l in ("bce", "focal", "wbce")]
    found = [(l, r) for l, r in found if r]
    if not found:
        continue
    abl += [f"### {ds.upper()}", "", "| loss | best epoch | val dice | test dice | test iou | test acc | test sens | test spec |",
            "|---|---|---|---|---|---|---|---|"]
    for l, r in found:
        t = r["test"]
        abl.append(f"| {l} | {r['best_epoch']} | {r['val']['dice']:.4f} | {t['dice']:.4f} | {t['iou']:.4f} | "
                   f"{t['acc']:.4f} | {t['sens']:.4f} | {t['spec']:.4f} |")
    abl.append("")
(RES / "ablation_loss.md").write_text("\n".join(abl))
print("\n".join(abl))
