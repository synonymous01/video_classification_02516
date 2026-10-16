#!/bin/sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J 3dcnns_groupISA
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"

#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

source /zhome/de/b/233869/video_class/bin/activate
python cnns.py
