#!/bin/sh 

##### k2, s0, d, v7, active_v8, numx, numy
python shell_bending_batch.py 0.2 400 10e-4 0.01 1 200 200 
python shell_bending_batch.py 0.1 400 10e-4 0.01 1 200 200 
python shell_bending_batch.py 0.1 400 10e-4 0.005 1 200 200 
python shell_bending_batch.py 0.2 400 10e-4 0.005 1 200 200 