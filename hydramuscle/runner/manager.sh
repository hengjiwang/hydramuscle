#!/bin/sh

python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.08 --s0=60 --v_delta=0.05 & \
python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.08 --s0=100 --v_delta=0.03 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.1 --s0=100 --v_delta=0.02 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=400 --v_delta=0.02 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=200 --v_delta=0.04 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.02 --s0=400 --v_delta=0.04 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.1 --s0=100 --v_delta=0.03 & \
# python layer_runner.py --gip3x=10 --gip3y=10 --k_ipr=0.01 --s0=100 --v_delta=0.03

