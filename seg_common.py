"""Shared code for the DRIVE (vessels) and PH2 (skin lesion) segmentation experiments.

Used by drive_cnn.py, drive_unet.py, ph2_cnn.py, ph2_unet.py (and the optional drive_cnn_focal.py / drive_cnn_bce.py).
Every experiment writes its own folder  results/<name>/  containing
  curves.png (loss / accuracy / Dice for train, val AND test), predictions.png,
  metrics.json, history.json, best.pt

Test curves are computed every epoch for DISPLAY ONLY. The weights that are kept
(and reported) are chosen by the best VALIDATION Dice.
Note: the loss differs between experiments, so loss values are not comparable across them.
"""
import os, random, time, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                      # no display on the GPU node
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- config (can be overridden with environment variables) ----------------
BASE = Path("/dtu/datasets1/02516")                 # DTU-HPC course datasets
_size = os.environ.get("SIZE")                      # SIZE=256 overrides the size of BOTH datasets
SIZES = {"drive": int(_size or 512), "ph2": int(_size or 256)}   # images resized to SIZE x SIZE
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
EPOCHS = int(os.environ.get("EPOCHS", 100))
LR = float(os.environ.get("LR", 1e-3))
SEED = int(os.environ.get("SEED", 42))
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


# ---------------- data ----------------
def find_root(dataset):
    if dataset == "drive":
        env, marker = "DRIVE_ROOT", "1st_manual"
        cands = [BASE / "DRIVE" / "training", Path("driver/training"), Path("DRIVE/training")]
    else:
        env, marker = "PH2_ROOT", "IMD002"
        cands = [BASE / "PH2_Dataset_images", BASE / "PH2", Path("PH2_Dataset_images")]
    if os.environ.get(env):
        cands.insert(0, Path(os.environ[env]))
    for c in cands:
        if (c / marker).exists():
            return c
    if BASE.exists():                                # search a few levels below the course folder
        for pat in ("*/", "*/*/", "*/*/*/"):
            for hit in sorted(BASE.glob(pat + marker)):
                return hit.parent
    print(f"{dataset} folder not found. Set {env}=<folder>. Contents of {BASE}:",
          sorted(os.listdir(BASE)) if BASE.exists() else "folder missing")
    raise FileNotFoundError(dataset)


def make_split(dataset, root):
    if dataset == "drive":                           # 20 labeled images -> 12 / 4 / 4
        ids = sorted(int(f.split("_")[0]) for f in os.listdir(root / "images") if f.endswith(".tif"))
        assert len(ids) == 20, len(ids)
        n_tr, n_va = 12, 4
    else:                                            # 200 cases -> 140 / 30 / 30
        ids = sorted(d for d in os.listdir(root) if d.startswith("IMD"))
        assert len(ids) == 200, len(ids)
        n_tr, n_va = 140, 30
    random.Random(SEED).shuffle(ids)
    return sorted(ids[:n_tr]), sorted(ids[n_tr:n_tr + n_va]), sorted(ids[n_tr + n_va:])


def load_one(dataset, root, i, size):
    if dataset == "drive":
        img = Image.open(root / "images" / f"{i}_training.tif")
        lab = Image.open(root / "1st_manual" / f"{i}_manual1.gif")      # vessels
        fov = Image.open(root / "mask" / f"{i}_training_mask.gif")      # valid region, NOT vessels
    else:
        img = Image.open(root / i / f"{i}_Dermoscopic_Image" / f"{i}.bmp")
        lab = Image.open(root / i / f"{i}_lesion" / f"{i}_lesion.bmp")  # lesion mask
        fov = None
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    lab = lab.convert("L").resize((size, size), Image.NEAREST)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)
    y = torch.from_numpy((np.asarray(lab) > 127).astype(np.float32))[None]
    if fov is None:
        m = torch.ones_like(y)                       # PH2 has no FOV mask: use all pixels
    else:
        m = torch.from_numpy((np.asarray(fov.convert("L").resize((size, size), Image.NEAREST)) > 127).astype(np.float32))[None]
    return x, y, m


def load_split(dataset, root, id_list, size):
    xs, ys, ms = zip(*[load_one(dataset, root, i, size) for i in id_list])
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


# ---------------- models ----------------
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EncDec(nn.Module):
    """Simple encoder-decoder CNN: 3 down, 3 up, no skip connections."""
    def __init__(self, c=(16, 32, 64, 128)):
        super().__init__()
        self.e1, self.e2, self.e3, self.bott = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), block(c[2], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), block(c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), block(c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(self.pool(x))
        x = self.e3(self.pool(x))
        x = self.bott(self.pool(x))
        x = self.d3(self.u3(x))
        x = self.d2(self.u2(x))
        x = self.d1(self.u1(x))
        return self.out(x)


class UNet(nn.Module):
    """U-Net: 4 down, 4 up, with skip connections. SIZE must be divisible by 16."""
    def __init__(self, c=(16, 32, 64, 128), cb=256):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
        self.bott = block(c[3], cb)
        self.pool = nn.MaxPool2d(2)
        self.u4, self.d4 = nn.ConvTranspose2d(cb, c[3], 2, stride=2), block(2 * c[3], c[3])
        self.u3, self.d3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), block(2 * c[2], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), block(2 * c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), block(2 * c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.bott(self.pool(s4))
        x = self.d4(torch.cat([self.u4(b), s4], 1))
        x = self.d3(torch.cat([self.u3(x), s3], 1))
        x = self.d2(torch.cat([self.u2(x), s2], 1))
        x = self.d1(torch.cat([self.u1(x), s1], 1))
        return self.out(x)


MODELS = {"cnn": EncDec, "unet": UNet}


# ---------------- losses (all only count pixels inside the FOV) ----------------
def make_loss(name, pos_weight):
    if name == "bce":            # plain cross entropy
        def f(lg, y, m):
            l = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
            return (l * m).sum() / m.sum()
    elif name == "wbce":         # cross entropy with positive-class weight
        def f(lg, y, m):
            l = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_weight, reduction="none")
            return (l * m).sum() / m.sum()
    elif name == "focal":        # focal loss, gamma=2, alpha=0.75 (more weight on the rare vessel class)
        def f(lg, y, m, gamma=2.0, alpha=0.75):
            b = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
            p = torch.sigmoid(lg)
            pt = p * y + (1 - p) * (1 - y)
            a = alpha * y + (1 - alpha) * (1 - y)
            return ((a * (1 - pt) ** gamma * b) * m).sum() / m.sum()
    else:
        raise ValueError(name)
    return f


# ---------------- metrics ----------------
def counts(logits, y, m):
    p = (torch.sigmoid(logits) > 0.5).float()
    tp = (p * y * m).sum().item(); fp = (p * (1 - y) * m).sum().item()
    fn = ((1 - p) * y * m).sum().item(); tn = ((1 - p) * (1 - y) * m).sum().item()
    return np.array([tp, fp, fn, tn])


def metrics_from(c, eps=1e-8):
    tp, fp, fn, tn = c
    return dict(dice=2 * tp / (2 * tp + fp + fn + eps), iou=tp / (tp + fp + fn + eps),
                acc=(tp + tn) / (tp + fp + fn + tn + eps),
                sens=tp / (tp + fn + eps), spec=tn / (tn + fp + eps))


@torch.no_grad()
def evaluate(model, lossfn, X, Y, M):
    model.eval(); loss_sum, c = 0.0, np.zeros(4)
    for i in range(0, len(X), BATCH_SIZE):
        x, y, m = X[i:i+BATCH_SIZE].to(device), Y[i:i+BATCH_SIZE].to(device), M[i:i+BATCH_SIZE].to(device)
        lg = model(x)
        loss_sum += lossfn(lg, y, m).item() * len(x)
        c += counts(lg, y, m)
    out = metrics_from(c); out["loss"] = loss_sum / len(X)
    return out


def augment(x, y, m):
    if random.random() < 0.5: x, y, m = x.flip(-1), y.flip(-1), m.flip(-1)
    if random.random() < 0.5: x, y, m = x.flip(-2), y.flip(-2), m.flip(-2)
    return x, y, m


# ---------------- one full experiment ----------------
def run(name, model_name, loss_name, dataset="drive"):
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    size = SIZES[dataset]
    root = find_root(dataset)
    print(f"=== {name}: dataset={dataset} model={model_name} loss={loss_name} device={device} size={size} epochs={EPOCHS} ===", flush=True)
    print("data folder:", root, flush=True)

    train_ids, val_ids, test_ids = make_split(dataset, root)
    print(f"split sizes: {len(train_ids)}/{len(val_ids)}/{len(test_ids)} | test ids: {test_ids}", flush=True)
    Xtr, Ytr, Mtr = load_split(dataset, root, train_ids, size)
    Xva, Yva, Mva = load_split(dataset, root, val_ids, size)
    Xte, Yte, Mte = load_split(dataset, root, test_ids, size)

    # normalization statistics from the TRAIN set only, applied to all splits
    mean = Xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = Xtr.std(dim=(0, 2, 3), keepdim=True)
    Xtr, Xva, Xte = [(x - mean) / std for x in (Xtr, Xva, Xte)]

    pos = (Ytr * Mtr).sum(); neg = ((1 - Ytr) * Mtr).sum()
    pos_weight = (neg / pos).to(device)
    print("foreground fraction in FOV: %.3f | pos_weight: %.1f" % ((pos / (pos + neg)).item(), pos_weight.item()), flush=True)

    model = MODELS[model_name]().to(device)
    lossfn = make_loss(loss_name, pos_weight)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    keys = ["loss", "acc", "dice"]
    hist = {f"{s}_{k}": [] for s in ("train", "val", "test") for k in keys}
    best_dice, best_ep, best_state = -1.0, 0, None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x, y, m = augment(Xtr[idx], Ytr[idx], Mtr[idx])
            x, y, m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = lossfn(model(x), y, m)
            loss.backward(); opt.step()

        res = {"train": evaluate(model, lossfn, Xtr, Ytr, Mtr),
               "val": evaluate(model, lossfn, Xva, Yva, Mva),
               "test": evaluate(model, lossfn, Xte, Yte, Mte)}      # test: display only
        for s in res:
            for k in keys: hist[f"{s}_{k}"].append(res[s][k])
        if res["val"]["dice"] > best_dice:                          # selection uses VAL only
            best_dice, best_ep = res["val"]["dice"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 5 == 0:
            print(f"ep {ep:3d} | loss tr {res['train']['loss']:.4f} va {res['val']['loss']:.4f} te {res['test']['loss']:.4f}"
                  f" | acc tr {res['train']['acc']:.4f} va {res['val']['acc']:.4f} te {res['test']['acc']:.4f}"
                  f" | dice tr {res['train']['dice']:.3f} va {res['val']['dice']:.3f} te {res['test']['dice']:.3f}"
                  f" | {time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, out / "best.pt")
    print(f"best val dice {best_dice:.4f} at epoch {best_ep} (weights restored)", flush=True)

    # ---- curves: loss, accuracy, Dice for train / val / test ----
    x_axis = range(1, EPOCHS + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for a, k, title in zip(ax, keys, [f"Loss ({loss_name})", "Accuracy (pixels inside FOV)", "Dice"]):
        a.plot(x_axis, hist[f"train_{k}"], label="train")
        a.plot(x_axis, hist[f"val_{k}"], label="val")
        a.plot(x_axis, hist[f"test_{k}"], "--", label="test (display only)")
        a.axvline(best_ep, color="gray", lw=0.8, ls=":", label=f"best val epoch ({best_ep})")
        a.set_title(f"{name}: {title}"); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(out / "curves.png", dpi=150); plt.close(fig)

    # ---- final metrics with the best-val weights ----
    final = {"name": name, "dataset": dataset, "model": model_name, "loss": loss_name, "best_epoch": best_ep, "size": size,
             "train": evaluate(model, lossfn, Xtr, Ytr, Mtr),
             "val": evaluate(model, lossfn, Xva, Yva, Mva),
             "test": evaluate(model, lossfn, Xte, Yte, Mte)}
    for s in ("train", "val", "test"):
        print(f"{s:5s} " + " | ".join(f"{k} {v:.4f}" for k, v in final[s].items()), flush=True)
    json.dump(final, open(out / "metrics.json", "w"), indent=2)
    json.dump(hist, open(out / "history.json", "w"))

    # ---- test predictions ----
    model.eval()
    with torch.no_grad():
        pred = torch.cat([(torch.sigmoid(model(Xte[i:i+BATCH_SIZE].to(device))) > 0.5).float().cpu()
                          for i in range(0, len(Xte), BATCH_SIZE)])
    n_show = min(len(Xte), 8)
    fig, ax = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    for r in range(n_show):
        img = (Xte[r] * std[0] + mean[0]).permute(1, 2, 0).clamp(0, 1)
        ax[r, 0].imshow(img); ax[r, 1].imshow(Yte[r, 0], cmap="gray"); ax[r, 2].imshow(pred[r, 0] * Mte[r, 0], cmap="gray")
        for a, t in zip(ax[r], [f"image {test_ids[r]}", "ground truth", "prediction"]):
            a.set_title(t); a.axis("off")
    plt.tight_layout(); plt.savefig(out / "predictions.png", dpi=120); plt.close(fig)
    print("saved to", out, flush=True)
