@echo off
D:
cd D:\hydra\hydra\runner
start python test_shell_random_slowwaves.py --T=50 --randomnum=20 --gip3x=10 --gip3y=10
start python test_shell_random_slowwaves.py --T=50 --randomnum=20 --gip3x=20 --gip3y=20
start python test_shell_random_slowwaves.py --T=50 --randomnum=10 --gip3x=10 --gip3y=10
start python test_shell_random_slowwaves.py --T=50 --randomnum=10 --gip3x=20 --gip3y=20
exit