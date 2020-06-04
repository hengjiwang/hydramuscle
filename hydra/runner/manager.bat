@echo off
D:
cd D:\hydra\hydra\runner
start python shell_runner.py --sparsity=0.1 --gc=1000 --gip3=5
start python shell_runner.py --sparsity=0.1 --gc=1000 --gip3=10
start python shell_runner.py --sparsity=0.05 --gc=1000 --gip3=5
exit