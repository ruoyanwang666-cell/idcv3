#!/bin/sh
# Main experiments, one job, one after another:
#   shape check -> DRIVE: CNN, U-Net -> PH2: CNN, U-Net -> summary tables
# Loss ablation (CE / focal) is a separate job: run_ablation.sh
# Submit from the folder that contains all the .py files with:   bsub < run_drive.sh
#BSUB -J seg_all
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 03:00
#BSUB -o seg_all_%J.out
#BSUB -e seg_all_%J.err

# change this to the environment that has torch with CUDA
source ~/venv_1/bin/activate

# run in the folder the job was submitted from
cd "$LSB_SUBCWD"
nvidia-smi

# stop early if a file was not uploaded
for f in seg_common.py unet_shapes.py drive_cnn.py drive_unet.py ph2_cnn.py ph2_unet.py summarize_results.py; do
    [ -f "$f" ] || { echo "MISSING FILE: $f (upload it to $PWD)"; exit 1; }
done
mkdir -p results

# U-Net sketch + shape check first (fast); stops the job if a shape does not match
python -u unet_shapes.py || exit 1

# optional quick test: remove the leading # to run only 3 epochs at low resolution
# export EPOCHS=3 SIZE=128

# each experiment writes to its own folder: results/<name>/  (a failure does not stop the others)
python -u drive_cnn.py  || echo "drive_cnn.py FAILED"
python -u drive_unet.py || echo "drive_unet.py FAILED"
python -u ph2_cnn.py    || echo "ph2_cnn.py FAILED"
python -u ph2_unet.py   || echo "ph2_unet.py FAILED"

python summarize_results.py
