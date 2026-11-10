"""DRIVE: encoder-decoder CNN, cross entropy with positive-class weight."""
from seg_common import run

run("drive_cnn", "cnn", "wbce", "drive")
