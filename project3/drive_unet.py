"""DRIVE: U-Net, cross entropy with positive-class weight."""
from seg_common import run

run("drive_unet", "unet", "wbce", "drive")
