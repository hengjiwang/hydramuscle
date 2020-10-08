@echo off
E:
cd E:\hydra\hydra\test
start python test_shell_cycles.py --T=500 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0 --save_interval=100 --sparsity=0.02 --k_deg=0.02 --k_ipr=0.2 --seed=1112
start python test_shell_cycles.py --T=500 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0 --save_interval=100 --sparsity=0.02 --k_deg=0.01 --k_ipr=0.2 --seed=1112
start python test_shell_cycles.py --T=500 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --v_delta=0 --save_interval=100 --sparsity=0.02 --k_deg=0.15 --k_ipr=0.2 --seed=1112