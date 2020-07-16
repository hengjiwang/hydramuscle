@echo off
D:
cd D:\hydra\hydra\runner
start python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=2.5 --gip3y=20 --gip3=2.5
start python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=2 --gip3y=20 --gip3=2
start python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=1 --gip3y=10 --gip3=1
start python test_shell_cycles.py --T=300 --numx=50 --numy=100 --gip3x=1.5 --gip3y=10 --gip3=1.5