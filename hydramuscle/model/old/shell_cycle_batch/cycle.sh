#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy, k9, gkca
python shell_cycle_ecto_batch.py 0.2 400 10e-4 0.01 1 200 200 0.08 10e-9 & \
python shell_cycle_ecto_batch.py 0.1 400 10e-4 0.01 1 200 200 0.08 10e-9 & \
python shell_cycle_ecto_batch.py 0.2 400 10e-4 0.005 1 200 200 0.08 10e-9 & \
python shell_cycle_ecto_batch.py 0.1 400 10e-4 0.005 1 200 200 0.08 10e-9
# python shell_cycle_endo_batch.py 0.05 200 10e-4 0.00 0.4 200 200 0.01 0
# python shell_cycle_endo_batch.py 0.05 200 10e-4 0.00 1 200 200 0.01 0
# python shell_cycle_batch.py 0.05 200 10e-4 0.00 0.05 200 200 0.02 0 & \
# python shell_cycle_batch.py 0.05 200 10e-4 0.00 0.10 200 200 0.02 0 & \
# python shell_cycle_batch.py 0.05 200 10e-4 0.00 0.20 200 200 0.02 0 