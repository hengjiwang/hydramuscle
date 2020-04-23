import modin.pandas as pd
import time
import pandas

start = time.time()
pd.read_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/c_200x200_100s_ele_bottom_200_conductance.csv')
print(time.time() - start)

start = time.time()
pandas.read_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/c_200x200_100s_ele_bottom_200_conductance.csv')
print(time.time() - start)