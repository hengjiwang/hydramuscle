#!/bin/sh

python test_density_slow.py 0.0001 & \
python test_density_slow.py 0.0002 & \
python test_density_slow.py 0.0005 & \
python test_density_slow.py 0.00075 & \
python test_density_slow.py 0.001 & \
python test_density_slow.py 0.002 & \
python test_density_slow.py 0.005 & \
python test_density_slow.py 0.0075 & \
python test_density_slow.py 0.01 & \
python test_density_slow.py 0.02 & \
python test_density_slow.py 0.05 & \
python test_density_slow.py 0.075 & \
python test_density_slow.py 0.1 & \
python test_density_slow.py 0.2 & \
python test_density_slow.py 0.5 & \