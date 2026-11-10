"""Loss ablation: U-Net on DRIVE, only the loss changes (focal)."""
from seg_common import run

run("drive_unet_focal", "unet", "focal", "drive")
