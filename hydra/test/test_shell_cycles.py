import sys, os, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import argparse
import random

from hydra.model.smc import SMC
from hydra.model.layer import Layer
from hydra.model.shell import Shell

def run_shell(numx, numy, seed, gip3x, gip3y, gcx=1000, gcy=1000, sparsity=0.002, gc=1000, gip3=5, pathway="Both",
              save_interval=100, save_dir="../results/data/calcium", **kargs):
    
    # Check the existence of the saving directory
    if not save_dir.endswith('/'):
        save_dir += '/'
    if not os.path.isdir(save_dir):
        raise FileExistsError("Target directory " + save_dir + " does not exist. ")

    # Build layers
    smc = SMC(**kargs)
    ectoderm = Layer(smc, numx=numx, numy=numy, gip3x=gip3x, gip3y=gip3y,
                     gcx=gcx, gcy=gcy, save_interval=save_interval)
    endoderm = Layer(smc, numx=numx, numy=numy, gip3x=gip3x, gip3y=gip3y,
                     gcx=gcx, gcy=gcy, save_interval=save_interval)
    
    # Define stimulation times
    # stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7]
    # stims_slow1 = [14]
    # stims_slow2 = []
    init1 = 100
    init2 = 223
    # stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
    #               init1, init1 + 5.2, init1 + 8.2, init1 + 10.6, init1 + 12.8, init1 + 15, init1 + 17.3, init1 + 19.4, init1 + 21.9, init1 + 25.1, init1 + 29.5, init1 + 34.3,
    #               init2, init2 + 5.7, init2 + 8.8, init2 + 11.6, init2 + 13.8, init2 + 16.1, init2 + 18.3, init2 + 21, init2 + 24.2, init2 + 29, init2 + 35.4]
    # stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
    #             init1, init1 + 4.75, init1 + 7.75, init1 + 9.25, init1 + 13.75, init1 + 17.5, init1 + 21.5, init1 + 24.5, init1 + 30,
    #             init2, init2 + 4.25, init2 + 7, init2 + 8.75, init2 + 11.5, init2 + 13.25, init2 + 17, init2 + 18.75, init2 + 22.75, init2 + 27, init2 + 29.25]
    # stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
    #         init1, init1 + 4.75, init1 + 9.25, init1 + 13.75, init1 + 17.5, init1 + 21.5, init1 + 24.5, init1 + 30,
    #         init2, init2 + 4.25, init2 + 8.75, init2 + 13.25, init2 + 17, init2 + 22.75, init2 + 29.25]
    # stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
    #               init1+0.0, init1+9.25, init1+14.5, init1+18.25, init1+21.5, init1+24.5, init1+28.0, init1+31.5, init1+35.25, init1+40.5,
    #               init2+0.0, init2+7.75, init2+15.25, init2+19.75, init2+23.0, init2+27.25, init2+30.5, init2+34.25, init2+38.25, init2+43.5]
    stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
                  100.0, 109.5, 115.5, 118.5, 122.25, 125.75, 129.0, 132.5, 136.0, 143.25,
                  208.0, 209.75, 217.75, 223.75, 227.25, 230.25, 233.75, 237.5, 242.0, 249.75,
                  339.75, 347.25, 352.5, 356.25, 359.75, 363.0, 366.25, 370.0, 375.25,
                  462.25, 471.5, 476.75, 480.5, 483.75, 486.75, 490.25, 493.75, 497.5, 502.75]
    stims_slow1 = []
    stims_slow2 = []
    # stims_slow1 = [15]
    # stims_slow2 = []

    # Set stimulation patterns
    if pathway == "Both":
        ectoderm.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
        ectoderm.set_stim_pattern("slow", xmin=13, xmax=17, ymin=0, ymax=4,
                                  stim_times=stims_slow1)
        ectoderm.set_stim_pattern("slow", xmin=0, xmax=2, ymin=0, ymax=4,
                                  stim_times=stims_slow2)
        ectoderm.set_stim_pattern("slow", xmin=numx-2, xmax=numx, ymin=0, ymax=4,
                                  stim_times=stims_slow2)
    elif pathway == "Fast":
        ectoderm.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
    elif pathway == "Slow":
        ectoderm.set_stim_pattern("slow", xmin=13, xmax=17, ymin=0, ymax=4,
                                  stim_times=stims_slow1)

    # Build shell
    shell = Shell(ectoderm, endoderm, seed, sparsity, gc, gip3)

    # Run the model
    sol = shell.run()
    sol = pd.DataFrame(sol)

    # Generate filename and corresponding metadata
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    filemeta = "numx=" + str(numx) + ","
    filemeta += "numy=" + str(numy) + ","
    filemeta += "gc=" + str(gc) + ","
    filemeta += "gip3=" + str(gip3) + ","
    filemeta += "gip3x=" + str(gip3x) + ","
    filemeta += "gip3y=" + str(gip3y) + ","
    filemeta += "sparsity=" + str(sparsity) + ","
    filemeta += "seed=" + str(seed)

    for key in kargs:
        filemeta += "," + key + '=' + str(kargs[key])

    filemeta += ",shell"

    # Save the results
    sol.to_hdf(save_dir + filename + '.h5', 'calcium')

    # Document the metadata
    with open("../results/data/meta.txt", "a+") as metafile:
        metafile.write(filename + "    " + filemeta + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--numx', type=int, default=200)
    parser.add_argument('--numy', type=int, default=200)
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000))
    parser.add_argument('--gip3x', type=float, default=5)
    parser.add_argument('--gip3y', type=float, default=40)
    parser.add_argument('--gcx', type=float, default=1000)
    parser.add_argument('--gcy', type=float, default=1000)
    parser.add_argument('--sparsity', type=float, default=0.002)
    parser.add_argument('--gc', type=float, default=1000)
    parser.add_argument('--gip3', type=float, default=5)
    parser.add_argument('--pathway', type=str, default="Both")
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="../results/data/calcium")
    parser.add_argument('--T', type=float, default=100)
    parser.add_argument('--dt', type=float, default=0.0002)
    parser.add_argument('--k_ipr', type=float, default=0.2)
    parser.add_argument('--s0', type=float, default=100)
    parser.add_argument('--d', type=float, default=20e-4)
    parser.add_argument('--v_delta', type=float, default=0.04)
    parser.add_argument('--k_deg', type=float, default=0.15)
    args = parser.parse_args()

    run_shell(args.numx, args.numy, args.seed, args.gip3x, args.gip3y,
              args.gcx, args.gcy, args.sparsity, args.gc, args.gip3, 
              args.pathway, args.save_interval, args.save_dir,
              T=args.T, dt=args.dt, k_ipr=args.k_ipr, s0=args.s0,
              d=args.d, v_delta=args.v_delta, k_deg=args.k_deg)




    


    