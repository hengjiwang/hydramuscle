#!/bin/sh

python test_shell_cycles.py --save_interval=500 --T=300 --v_delta=0.03
python test_shell_cycles.py --save_interval=500 --T=300 --v_delta=0.03 --k_ipr=0.15
python test_shell_cycles.py --save_interval=500 --T=300 --v_delta=0.03 --k_ipr=0.1