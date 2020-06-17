@echo off
D:
cd D:\hydra\hydra\runner
start python test_shell_cycles.py --T=100 --sparsity=0.002
start python test_shell_cycles.py --T=100 --sparsity=0.001
start python test_shell_cycles.py --T=100 --sparsity=0.0005
exit