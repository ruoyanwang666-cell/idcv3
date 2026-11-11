"""Part 2: PH2, U-Net, weak supervision with simulated point clicks.

Usage:  python ph2_points.py <positive clicks> <negative clicks> [random|center|edge]
Example: python ph2_points.py 5 5 center   ->  results/ph2_pts_center_p5_n5/

Same U-Net, split, seed, optimiser and epochs as the fully supervised run ph2_unet_bce;
the only change is that the cross-entropy loss is computed on the clicked pixels only.
"""
import sys
from seg_common import run

pos, neg = int(sys.argv[1]), int(sys.argv[2])
strategy = sys.argv[3] if len(sys.argv) > 3 else "random"
run(f"ph2_pts_{strategy}_p{pos}_n{neg}", "unet", "bce", "ph2",
    clicks=dict(pos=pos, neg=neg, strategy=strategy))
