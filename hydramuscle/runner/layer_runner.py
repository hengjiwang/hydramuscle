import sys, os, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import argparse
import h5py

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
    stims_fast = [0, 5.2, 8.2, 10.6, 12.8, 15, 17.3, 19.4, 21.9, 25.1, 29.5, 34.3]
    stims_slow = [14]

    # Set stimulation patterns
    if pathway == "Both":
        layer.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
        layer.set_stim_pattern("slow", xmin=90, xmax=110, ymin=0, ymax=10,
                            stim_times=stims_slow)
    elif pathway == "Fast":
        layer.set_stim_pattern("fast", xmin=0, xmax=numx, ymin=0, ymax=1, stim_times=stims_fast)
    elif pathway == "Slow":
        layer.set_stim_pattern("slow", xmin=90, xmax=110, ymin=0, ymax=10,
                            stim_times=stims_slow)

    # Run the model
    sol = layer.run()
    sol = pd.DataFrame(sol)

    # Generate filename and corresponding metadata
    filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    filemeta = "numx=" + str(numx) + ","
    filemeta += "numy=" + str(numy)

    for key in kargs:
        filemeta += "," + key + '=' + str(kargs[key])

    # filemeta += ",slow"

    # Save the results
    # hf = h5py.File(save_dir + filename + '.h5', 'w')
    # hf.create_dataset('calcium', data=sol)
    # hf.close()
    sol.to_hdf(save_dir + filename + '.h5', 'calcium')

    # Document the metadata
    with open(save_dir+"meta.txt", "a+") as metafile:
        metafile.write(filename + "    " + filemeta + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--numx', type=int, default=200)
    parser.add_argument('--numy', type=int, default=200)
    parser.add_argument('--gip3x', type=float, default=None)
    parser.add_argument('--gip3y', type=float, default=None)
    parser.add_argument('--gcx', type=float, default=1000)
    parser.add_argument('--gcy', type=float, default=1000)
    parser.add_argument('--pathway', type=str, default="Both")
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default="../results/data/calcium")
    parser.add_argument('--T', type=float, default=100)
    parser.add_argument('--dt', type=float, default=0.0002)
    parser.add_argument('--k_ipr', type=float, default=None)
    parser.add_argument('--s0', type=float, default=None)
    parser.add_argument('--d', type=float, default=20e-4)
    parser.add_argument('--v_delta', type=float, default=None)
    args = parser.parse_args()

    run_layer(args.numx, args.numy, args.gip3x, args.gip3y,
              args.gcx, args.gcy, args.pathway, args.save_interval, args.save_dir,
              T=args.T, dt=args.dt, k_ipr=args.k_ipr, s0=args.s0,
              d=args.d, v_delta=args.v_delta)
