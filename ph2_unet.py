"""PH2: U-Net, cross entropy with positive-class weight."""
from seg_common import run

run("ph2_unet", "unet", "wbce", "ph2")
