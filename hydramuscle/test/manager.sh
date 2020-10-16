#!/bin/sh
#python test_shell_cycles.py --T=50 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --save_interval=100 --sparsity=0.2 --k_deg=0.05 --k_ipr=0.2 & \
#python test_shell_cycles.py --T=50 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --save_interval=100 --sparsity=0.02 --k_deg=0.05 --k_ipr=0.2 & \
#python test_shell_cycles.py --T=50 --numx=30 --numy=60 --gip3x=0.1 --gip3y=2 --gip3=0.1 --save_interval=100 --sparsity=0.0005 --k_deg=0.05 --k_ipr=0.2

python test_density.py 0.0001 & \
python test_density.py 0.0002 & \
python test_density.py 0.0005 & \
python test_density.py 0.00075 & \
python test_density.py 0.001 & \
python test_density.py 0.002 & \
python test_density.py 0.005 & \
python test_density.py 0.0075 & \
python test_density.py 0.01 & \
python test_density.py 0.02 & \
python test_density.py 0.05 & \
python test_density.py 0.075 & \
python test_density.py 0.1 & \
python test_density.py 0.2 & \
python test_density.py 0.5