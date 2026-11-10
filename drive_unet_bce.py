"""Loss ablation: U-Net on DRIVE, only the loss changes (bce)."""
from seg_common import run

run("drive_unet_bce", "unet", "bce", "drive")
