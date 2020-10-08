import os
import datetime
import pandas as pd
import argparse

from hydramuscle.model.smc import SMC
from hydramuscle.model.layer import Layer

def run_layer(numx, numy, gip3x, gip3y, gcx=1000, gcy=1000, pathway="Both",
              save_interval=100, save_dir="../results/data/calcium", **kargs):
    "Run the Layer"

    # Check the existence of the saving directory
    if not save_dir.endswith('/'):
        save_dir += '/'  
    if not os.path.isdir(save_dir):
        raise FileExistsError("Target directory " + save_dir + " does not exist. ")

    # Initialize model
    smc = SMC(**kargs)
    layer = Layer(smc, numx=numx, numy=numy, gip3x=gip3x, gip3y=gip3y,
                  gcx=gcx, gcy=gcy, save_interval=save_interval)

    # Define stimulation times
    stims_fast = [0, 4.4, 7.1, 9.3, 11.2, 13.2, 15.7, 18.4, 21.2, 24.7,
                  100, 105.2, 108.2, 110.6, 112.8, 115, 117.3, 119.4, 121.9, 125.1, 129.5, 134.3,
                  200, 205.7, 208.8, 211.6, 213.8, 216.1, 218.3, 221, 224.2, 229, 235.4]
    # stims_slow1 = [114]
    stims_slow1 = [14]
    stims_slow2 = [214]

    # Set stimulation patterns
    if pathway == "Both":
        layer.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
        layer.set_stim_pattern("slow", xmin=90, xmax=110, ymin=0, ymax=10,
                               stim_times=stims_slow1)
        layer.set_stim_pattern("slow", xmin=0, xmax=10, ymin=0, ymax=10,
                               stim_times=stims_slow2)
        layer.set_stim_pattern("slow", xmin=numx-10, xmax=numx, ymin=0, ymax=10,
                               stim_times=stims_slow2)
    elif pathway == "Fast":
        layer.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
    elif pathway == "Slow":
        layer.set_stim_pattern("slow", xmin=90, xmax=110, ymin=0, ymax=10,
                               stim_times=stims_slow1)

    # Run the model
    sol = layer.run()
    sol = pd.DataFrame(sol)

    # Generate filename and corresponding metadata
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    filemeta = "numx=" + str(numx) + ","
    filemeta += "numy=" + str(numy) + ","
    filemeta += "gip3x=" + str(gip3x) + ","
    filemeta += "gip3y=" + str(gip3y) + ","
    filemeta += "gcx=" + str(gcx)

    for key in kargs:
        filemeta += "," + key + '=' + str(kargs[key])

    if pathway == "Fast":
        filemeta += ",endo"

    # Save the results
    sol.to_hdf(save_dir + filename + '.h5', 'calcium')

    # Document the metadata
    with open("../results/data/meta.txt", "a+") as metafile:
        metafile.write(filename + "    " + filemeta + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--numx', type=int, default=200)
    parser.add_argument('--numy', type=int, default=200)
    parser.add_argument('--gip3x', type=float, default=5)
    parser.add_argument('--gip3y', type=float, default=40)
    parser.add_argument('--gcx', type=float, default=1000)
    parser.add_argument('--gcy', type=float, default=1000)
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

    run_layer(args.numx, args.numy, args.gip3x, args.gip3y,
              args.gcx, args.gcy, args.pathway, args.save_interval, args.save_dir,
              T=args.T, dt=args.dt, k_ipr=args.k_ipr, s0=args.s0,
              d=args.d, v_delta=args.v_delta, k_deg=args.k_deg)
