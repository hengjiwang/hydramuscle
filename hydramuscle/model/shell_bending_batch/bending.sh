#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy
python shell_bending_batch.py 0.2 400 40e-4 0.01 1 100 100 & \
python shell_bending_batch.py 0.2 500 40e-4 0.01 1 100 100 & \
python shell_bending_batch.py 0.2 200 40e-4 0.02 1 100 100