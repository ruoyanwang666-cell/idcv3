"""Loss ablation: U-Net on PH2, only the loss changes (bce)."""
from seg_common import run

run("ph2_unet_bce", "unet", "bce", "ph2")
