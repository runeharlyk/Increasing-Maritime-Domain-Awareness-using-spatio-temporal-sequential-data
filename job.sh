#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -q gpua100                       # queue/partition (check if a GPU queue is required)
#BSUB -W 02:00                      # wall-time hh:mm
#BSUB -n 4                         # CPU cores
#BSUB -R "span[hosts=1]"           # keep all cores on one node
#BSUB -R "select[gpu40gb]"
#BSUB -R "rusage[mem=8GB]"         # 4 GB RAM per core  → 16 GB total
#BSUB -gpu "num=1:mode=exclusive_process"   # ← *add* if you need one A100
#BSUB -u s234834@dtu.dk            # where e-mails go
#BSUB -B                           # e-mail at start   (optional)
#BSUB -N                           # e-mail at end     (optional)
#BSUB -oo logs/%J.out              # stdout  (overwrite)
#BSUB -eo logs/%J.err              # stderr  (overwrite)
# -------------------------------------------------

# stop on first error
set -e

# ---- software environment -----
module load cuda/11.8 # if your code needs the CUDA module

source .venv/bin/activate
              
nvidia-smi

# ---- train model ---------
python train.py