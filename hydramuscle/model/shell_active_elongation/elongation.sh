#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy, k9, gkca
python shell_elongation_batch.py 0.05 400 10e-4 0.01 1 200 200 0.01 0 &\
python shell_elongation_batch.py 0.05 400 10e-4 0.00 1 200 200 0.01 0 &\
python shell_elongation_batch.py 0.05 400 10e-4 0.01 1 200 200 0.005 0 &\
python shell_elongation_batch.py 0.05 400 10e-4 0.00 1 200 200 0.005 0