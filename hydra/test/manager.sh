#!/bin/sh
python test_shell_cycles.py --T=100 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0.03 --save_interval=100 --sparsity=0.2 --seed=1112 &\
python test_shell_cycles.py --T=100 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0.03 --save_interval=100 --sparsity=0.02 --seed=1112 &\
python test_shell_cycles.py --T=100 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0.03 --save_interval=100 --sparsity=0.00005 --seed=1112