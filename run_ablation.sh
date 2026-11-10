#!/bin/sh
# Loss ablation (U-Net only, the loss is the only change):
#   DRIVE and PH2 with plain cross entropy (bce) and focal loss.
#   The third loss (weighted CE) is the main U-Net run from run_drive.sh (drive_unet, ph2_unet).
# Submit from the folder that contains all the .py files with:   bsub < run_ablation.sh
#BSUB -J seg_ablation
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 03:00
#BSUB -o seg_ablation_%J.out
#BSUB -e seg_ablation_%J.err

# change this to the environment that has torch with CUDA
source ~/venv_1/bin/activate

# run in the folder the job was submitted from
cd "$LSB_SUBCWD"
nvidia-smi

for f in seg_common.py drive_unet_bce.py drive_unet_focal.py ph2_unet_bce.py ph2_unet_focal.py summarize_results.py; do
    [ -f "$f" ] || { echo "MISSING FILE: $f (upload it to $PWD)"; exit 1; }
done
mkdir -p results

# optional quick test: remove the leading # to run only 3 epochs at low resolution
# export EPOCHS=3 SIZE=128

python -u drive_unet_bce.py   || echo "drive_unet_bce.py FAILED"
python -u drive_unet_focal.py || echo "drive_unet_focal.py FAILED"
python -u ph2_unet_bce.py     || echo "ph2_unet_bce.py FAILED"
python -u ph2_unet_focal.py   || echo "ph2_unet_focal.py FAILED"

# rebuilds results/summary.md and results/ablation_loss.md from every results/*/metrics.json
# (the weighted-CE rows are filled in once the main job run_drive.sh has finished as well)
python summarize_results.py
