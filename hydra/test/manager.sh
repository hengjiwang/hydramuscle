#!/bin/sh
python test_shell_cycles.py --T=600 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --save_interval=100 --sparsity=0.02 --k_deg=0.05 --k_ipr=0.2 --seed=1112 & \
python test_shell_cycles.py --T=600 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --save_interval=100 --sparsity=0.02 --k_deg=0.02 --k_ipr=0.2 --seed=1112 