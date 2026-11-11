#!/bin/sh
# Part 2: weakly supervised PH2 segmentation with simulated point clicks (U-Net, same setup as Part 1).
# Ablation over the number of clicks (centre-biased sampling). 8 runs, about 2 minutes each.
# Submit from the folder that contains all the .py files with:   bsub < run_weak.sh
#BSUB -J seg_weak
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 02:00
#BSUB -o seg_weak_%J.out
#BSUB -e seg_weak_%J.err

# change this to the environment that has torch with CUDA
source ~/venv_1/bin/activate

# run in the folder the job was submitted from
cd "${LSB_SUBCWD:-$PWD}"
nvidia-smi

for f in seg_common.py ph2_points.py summarize_weak.py; do
    [ -f "$f" ] || { echo "MISSING FILE: $f (upload it to $PWD)"; exit 1; }
done
mkdir -p results

# optional quick test: remove the leading # to run only 3 epochs at low resolution
# export EPOCHS=3 SIZE=128

# 1) number of clicks (same number of positive and negative), centre-biased sampling
for n in 2 5 10 20 50 100; do
    python -u ph2_points.py $n $n center || echo "ph2_points.py $n $n center FAILED"
done

# 2) unbalanced clicks: mostly positive vs mostly negative (same total of 25 clicks)
python -u ph2_points.py 20 5 center || echo "ph2_points.py 20 5 center FAILED"
python -u ph2_points.py 5 20 center || echo "ph2_points.py 5 20 center FAILED"

# table of all point-supervision runs, with the fully supervised U-Net (ph2_unet_bce) as reference
python summarize_weak.py
