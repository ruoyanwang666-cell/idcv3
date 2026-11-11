#!/bin/sh
# Part 2, Task 4: sampling strategies. Repeats the click ablation with RANDOM sampling
# (uniform inside and outside the mask, the strategy the assignment suggests), so it can be
# compared with the centre-biased runs from run_weak.sh at every number of clicks.
# 6 runs, about 2 minutes each.
#
# Batch:        bsub < run_strategies.sh
# Interactive:  export CUDA_VISIBLE_DEVICES=<free gpu>;  bash run_strategies.sh 2>&1 | tee seg_strategies.out
#BSUB -J seg_strategies
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -W 02:00
#BSUB -o seg_strategies_%J.out
#BSUB -e seg_strategies_%J.err

# change this to the environment that has torch with CUDA
source ~/venv_1/bin/activate

# batch: the folder the job was submitted from; interactive: the current folder
cd "${LSB_SUBCWD:-$PWD}"
nvidia-smi

for f in seg_common.py ph2_points.py summarize_weak.py; do
    [ -f "$f" ] || { echo "MISSING FILE: $f (run this from the repo folder: $PWD)"; exit 1; }
done
mkdir -p results

# optional quick test: remove the leading # to run only 3 epochs at low resolution
# export EPOCHS=3 SIZE=128

# strategies to run. "edge" (clicks within 5 pixels of the lesion border) is optional:
# use  STRATEGIES="random edge"  to run it as well (6 more runs).
STRATEGIES="${STRATEGIES:-random}"

for s in $STRATEGIES; do
    for n in 2 5 10 20 50 100; do
        python -u ph2_points.py $n $n $s || echo "ph2_points.py $n $n $s FAILED"
    done
done

# rebuilds all three Part 2 tables in results/
python summarize_weak.py
