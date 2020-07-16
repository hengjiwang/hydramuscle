#!/bin/sh
# python test_shell_cycles.py --T=100 --numx=50 --numy=100 --gip3x=0.5 --gip3y=5 --gip3=0.5 & \
# python test_shell_cycles.py --T=100 --numx=50 --numy=100 --gip3x=0.2 --gip3y=2 --gip3=0.2 & \
# python test_shell_cycles.py --T=100 --numx=50 --numy=100 --gip3x=0.1 --gip3y=1 --gip3=0.1 & \
# python test_shell_cycles.py --T=100 --numx=50 --numy=100 --gip3x=0.2 --gip3y=2 --gip3=0.2 --v_delta=0.03 & \
# python test_shell_cycles.py --T=100 --numx=50 --numy=100 --gip3x=0.5 --gip3y=5 --gip3=0.5 --v_delta=0.03 & \
python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=0.05 --gip3y=2 --gip3=0.05 --v_delta=0.03 & \
python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=0.05 --gip3y=3 --gip3=0.05 --v_delta=0.03 & \
python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=0.05 --gip3y=4 --gip3=0.05 --v_delta=0.03 & \