#!/bin/bash
#PBS -P c07
#PBS -q normal
#PBS -l walltime=00:01:00
#PBS -l mem=96GB
#PBS -l jobfs=4GB
#PBS -l ncpus=24
#PBS -l wd

module load papi intel-compiler cmake/3.18.2 python3-as-python

python auto_test.py
