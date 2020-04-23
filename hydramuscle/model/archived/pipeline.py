import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from hydramuscle.model.smc import SMC
from hydramuscle.model.shell import Shell
from hydramuscle.postprocessing.force_encoder_2d import ForceEncoder2D
from hydramuscle.postprocessing.visualizer.save_video_parallel import *

TOTAL_TIME = 100
TIME_STEP = 0.0002
PARAM_K2 = 0.1
PARAM_S0 = 400
PARAM_D = 40E-4
PARAM_V7 = 0.01
BEHAVIOR = 'contraction burst'
NUMX = 200
NUMY = 200
STIMS_FAST = [1,3,5,7,9,12,15,18,22,26,31,36,42]
STIMS_SLOW = []

TARGET = "c_200x200_100s_ele_random_no_conductance"
NUMX = 200
NUMY = 200
TARGETFPS = 400

if __name__ == "__main__":

    # Run the model and save calcium data
    model = Shell(SMC(T=TOTAL_TIME, 
                      dt=TIME_STEP, 
                      k2=PARAM_K2, 
                      s0=PARAM_S0, 
                      d=PARAM_D, 
                      v7=PARAM_V7), 
                  behavior=BEHAVIOR, 
                  numx=NUMX, 
                  numy=NUMY)
    sol = model.run(STIMS_FAST, STIMS_SLOW)
    c = sol[:,0:model.numx*model.numy]
    df = pd.DataFrame(c)
    df.to_csv('/media/hengji/DATA/Data/Documents/hydramuscle/results/data/calcium/'+TARGET+'.csv', index = False)

    # Save the calcium data as frames and video
    if not os.path.exists('/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+TARGET+'/frames/'):
        os.makedirs('/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+TARGET+'/frames/')
        os.makedirs('/media/hengji/DATA/Data/Documents/hydramuscle/results/animations/'+TARGET+'/movie/')

    save_frames(c, TARGET, NUMX, NUMY)
    save_video(TARGET, TARGETFPS)

    # Encode calcium into force
    # ...