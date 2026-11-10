"""Loss ablation: U-Net on PH2, only the loss changes (focal)."""
from seg_common import run

run("ph2_unet_focal", "unet", "focal", "ph2")
