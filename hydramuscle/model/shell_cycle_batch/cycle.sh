#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy
python shell_cycle_batch.py 0.05 60 10e-4 0.02 1 200 200 & \
python shell_cycle_batch.py 0.05 60 10e-4 0.03 1 200 200 & \
python shell_cycle_batch.py 0.05 60 10e-4 0.04 1 200 200 & \
python shell_cycle_batch.py 0.05 100 10e-4 0.02 1 200 200 & \
python shell_cycle_batch.py 0.05 100 10e-4 0.03 1 200 200 & \
python shell_cycle_batch.py 0.05 100 10e-4 0.04 1 200 200