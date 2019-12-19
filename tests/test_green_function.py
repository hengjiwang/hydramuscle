from context import hydramuscle
import pandas as pd
from hydramuscle.model.shell import Shell
from hydramuscle.model.smc import SMC

def test_green_function():
    for scale in [10]:
        model = Shell(SMC(T=0.5, dt=0.0002, k2=0.1, s0=400, d=40e-4, v7=0),
                    'electrical points, '+str(scale)+'x', numx=50, numy=50)
        sol = model.run([0.1])
        df = pd.DataFrame(sol[:,4*model.num2:5*model.num2])
        df.to_csv('../results/data/voltage/v_50x50_05s_pulse_'+str(scale)+'x_both.csv', index = False)

if __name__ == "__main__":
    test_green_function()