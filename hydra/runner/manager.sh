#!/bin/sh

python layer_runner.py --gip3x=5 --gip3y=40 --k_ipr=0.3 --s0=60 --v_delta=0.04 --d=20e-4 --k_deg=0.15 & \
python layer_runner.py --gip3x=5 --gip3y=40 --k_ipr=0.3 --s0=100 --v_delta=0.04 --d=20e-4 --k_deg=0.15 & \
# python layer_runner.py --gip3x=5 --gip3y=40 --k_ipr=0.2 --s0=60 --v_delta=0.04 --d=20e-4 --k_deg=0.15 & \
# python layer_runner.py --gip3x=5 --gip3y=40 --k_ipr=0.2 --s0=100 --v_delta=0.04 --d=20e-4 --k_deg=0.15 & \
# python layer_runner.py --gip3x=5 --gip3y=30 --k_ipr=0.05 --s0=60 --v_delta=0.03 & \
# python layer_runner.py --gip3x=5 --gip3y=30 --k_ipr=0.05 --s0=100 --v_delta=0.03 & \
# python layer_runner.py --gip3x=5 --gip3y=20 --pathway=Fast --k_ipr=0.08 --s0=60 --v_delta=0 & \
# python layer_runner.py --gip3x=5 --gip3y=20 --pathway=Fast --k_ipr=0.04 --s0=100 --v_delta=0 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.1 --s0=100 --v_delta=0.02 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=400 --v_delta=0.02 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=200 --v_delta=0.04 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=400 --v_delta=0.04 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.1 --s0=100 --v_delta=0.03 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.01 --s0=100 --v_delta=0.03