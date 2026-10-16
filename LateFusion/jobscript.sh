#!/bin/sh
### -- Especificar la cola del curso --
#BSUB -q c02516
### -- Pedir una ranura de GPU (1 slice de 10GB) --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Nombre del trabajo --
#BSUB -J LateFusionJob
### -- Número de núcleos de CPU y memoria --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
### -- Tiempo máximo de ejecución (HH:MM) --
#BSUB -W 02:00
### -- Archivos de salida de texto (logs) --
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

### -- 1. Cargar el módulo de entorno virtual o python que uses --
# (Ajusta la ruta de tu entorno virtual si creaste uno)
source /zhome/3c/b/233219/venv_1/bin/activate
### -- 2. Ejecutar tu script de python principal --
python LateFusion.py