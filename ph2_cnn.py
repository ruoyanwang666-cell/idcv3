"""PH2: encoder-decoder CNN, cross entropy with positive-class weight."""
from seg_common import run

run("ph2_cnn", "cnn", "wbce", "ph2")
