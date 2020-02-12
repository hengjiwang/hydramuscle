#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy
python shell_cycle_batch.py 0.1 60 40e-4 0.04 10 50 50 & \
python shell_cycle_batch.py 0.1 60 40e-4 0.06 10 50 50 & \
python shell_cycle_batch.py 0.2 60 40e-4 0.06 10 50 50 & \
python shell_cycle_batch.py 0.1 60 40e-4 0.04 1 50 50 & \
python shell_cycle_batch.py 0.1 60 40e-4 0.06 1 50 50 & \
python shell_cycle_batch.py 0.2 60 40e-4 0.06 1 50 50