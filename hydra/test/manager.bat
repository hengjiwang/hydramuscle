@echo off
D:
cd D:\hydra\hydra\runner
start python test_shell_random_slowwaves.py --T=50 --randomnum=10
start python test_shell_random_slowwaves.py --T=50 --randomnum=20
start python test_shell_random_slowwaves.py --T=50 --randomnum=5
exit