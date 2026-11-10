"""U-Net shape check + sketch, for every input size used (DRIVE 512, PH2 256).

Builds the U-Net from seg_common, pushes a zero image through it, records the input and
output shape of every stage with forward hooks, and compares them with the expected sizes.
Exits with an error if any stage differs, and writes results/unet_architecture_<size>.png.
"""
import sys
import torch
import matplotlib
matplotlib.use("Agg")                      # no display on the GPU node
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from seg_common import UNet, RESULTS, SIZES

C = [16, 32, 64, 128]                      # encoder channels
BOTT = 256                                 # bottleneck channels


def expected(size):
    """(stage, output channels, output height = width, input channels) in the order of the sketch."""
    return [("e1", C[0], size, 3),
            ("e2", C[1], size // 2, C[0]),
            ("e3", C[2], size // 4, C[1]),
            ("e4", C[3], size // 8, C[2]),
            ("bott", BOTT, size // 16, C[3]),
            ("d4", C[3], size // 8, BOTT),            # input = up(bott) 128 ch + skip e4 128 ch
            ("d3", C[2], size // 4, 2 * C[2]),
            ("d2", C[1], size // 2, 2 * C[1]),
            ("d1", C[0], size, 2 * C[0]),
            ("out", 1, size, C[0])]


def check(size):
    assert size % 16 == 0, f"size {size} is not divisible by 16 (four poolings)"
    model = UNet().eval()
    seen, hooks = {}, []
    for name, *_ in expected(size):
        hooks.append(getattr(model, name).register_forward_hook(
            lambda mod, inp, out, n=name: seen.__setitem__(n, (tuple(inp[0].shape), tuple(out.shape)))))
    with torch.no_grad():
        y = model(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()

    print(f"\n=== U-Net shape check, input 3 x {size} x {size} ===")
    print(f"{'stage':6s} {'in channels':>12s} {'expected':>9s} | {'output (C,H,W)':>16s} {'expected':>16s} | ok")
    all_ok = True
    for name, c, s, cin in expected(size):
        ishape, oshape = seen[name]
        ok = (ishape[1] == cin) and (oshape == (1, c, s, s))
        all_ok &= ok
        print(f"{name:6s} {ishape[1]:12d} {cin:9d} | {str(oshape[1:]):>16s} {str((c, s, s)):>16s} | {'OK' if ok else 'MISMATCH'}")
    all_ok &= tuple(y.shape) == (1, 1, size, size)
    print("final output:", tuple(y.shape), "->", "OK" if all_ok else "MISMATCH")
    print("parameters:", sum(p.numel() for p in model.parameters()))
    return all_ok


def sketch(size):
    rows = expected(size)
    fig, ax = plt.subplots(figsize=(18, 6.5))
    colors = {"e": "#9fc5e8", "b": "#b6d7a8", "d": "#f4b6a0", "o": "#dddddd"}
    top = {}
    for i, (name, c, s, cin) in enumerate(rows):
        h = 0.5 + 3.5 * (s / size) ** 0.5
        ax.add_patch(Rectangle((i * 1.6, -h / 2), 1.0, h, fc=colors[name[0]], ec="black"))
        extra = f"\n(in: {cin})" if name.startswith("d") else ""
        ax.text(i * 1.6 + 0.5, 0, f"{c}\n{s}x{s}{extra}", ha="center", va="center", fontsize=9)
        ax.text(i * 1.6 + 0.5, h / 2 + 0.12, name, ha="center", fontweight="bold")
        top[i] = h / 2
        if i < len(rows) - 1:
            lab = "pool" if i < 4 else "up" if i < 8 else "1x1"
            ax.annotate("", xy=((i + 1) * 1.6 - 0.05, 0), xytext=(i * 1.6 + 1.05, 0),
                        arrowprops=dict(arrowstyle="->"))
            ax.text(i * 1.6 + 1.3, -0.3, lab, ha="center", fontsize=8)
    for k, (a, b) in enumerate([(0, 8), (1, 7), (2, 6), (3, 5)]):      # skip connections
        y = 2.6 + 0.35 * (3 - k)
        xa, xb = a * 1.6 + 0.5, b * 1.6 + 0.5
        ax.plot([xa, xa, xb, xb], [top[a] + 0.4, y, y, top[b] + 0.4], "--", color="gray", lw=1)
        ax.text((xa + xb) / 2, y + 0.05, "skip: concat", ha="center", fontsize=8, color="gray")
    ax.set_xlim(-0.3, len(rows) * 1.6)
    ax.set_ylim(-2.3, 3.9)
    ax.axis("off")
    ax.set_title(f"U-Net for {size}x{size} input (3 x {size} x {size}) - boxes show channels and "
                 f"feature-map size; each box = 2 x (conv3x3 + BN + ReLU)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"unet_architecture_{size}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("sketch saved to", out)


if __name__ == "__main__":
    good = True
    for s in sorted(set(SIZES.values())):
        ok = check(s)
        good &= ok
        sketch(s)
    sys.exit(0 if good else 1)
