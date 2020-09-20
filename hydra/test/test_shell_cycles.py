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
    init1 = 100
    init2 = 184.3
    # stims_fast = [0.0, 9.25, 14.5, 18.25, 21.5, 24.5, 28.0, 31.5, 35.25, 40.5,
    #               init1+0.0, init1+9.25, init1+14.5, init1+18.25, init1+21.5, init1+24.5, init1+28.0, init1+31.5, init1+35.25, init1+40.5,
    #               init2+0.0, init2+7.75, init2+15.25, init2+19.75, init2+23.0, init2+27.25, init2+30.5, init2+34.25, init2+38.25, init2+43.5]
    # stims_fast = [0.0, 9.5, 15.5, 18.5, 22.25, 25.75, 29.0, 32.5, 36.0, 43.25,
    #             100.0, 109.5, 115.5, 118.5, 122.25, 125.75, 129.0, 132.5, 136.0, 143.25,
    #             208.0, 209.75, 217.75, 223.75, 227.25, 230.25, 233.75, 237.5, 242.0, 249.75,
    #             339.75, 347.25, 352.5, 356.25, 359.75, 363.0, 366.25, 370.0, 375.25,
    #             462.25, 471.5, 476.75, 480.5, 483.75, 486.75, 490.25, 493.75, 497.5, 502.75]
    stims_fast = [0.0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
                 100.0, 104.4, 107.1, 109.3, 111.2, 113.2, 115.7, 118.4, 121.2, 124.7,
                 184.3, 188.0, 189.3, 190.4, 193.4, 195.0, 197.1, 199.4, 201.6, 204.8, 208.9, 215.8, 219.3, 222.2, 224.7, 227.6, 232.1,
                 284.7, 289.9, 292.9, 295.3, 297.5, 299.7, 302.0, 304.1, 306.6, 309.8, 314.2, 319.0,
                 376.4, 382.1, 385.2, 388.0, 390.2, 392.5, 394.7, 397.4, 400.6, 405.4, 411.8]

    init3 = 284.7
    init4 = 376.4

    stims_slow1 = [init1+20, init3+20]
    stims_slow2 = [init2+20, init4+20]

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