import sys, os, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import argparse
import random

from hydra.model.smc import SMC
from hydra.model.layer import Layer
from hydra.model.shell import Shell

def run_shell(numx, numy, seed, gip3x, gip3y, gcx=1000, gcy=1000, sparsity=0.002, gc=1000, gip3=5, pathway="Fast",
              save_interval=100, num_neurons=5, save_dir="../results/data/calcium", **kargs):

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
    stims_fast = [0.0, 5.0, 10.0]

    # Set stimulation patterns
    ped_ring = list(range(numx))
    neurons = random.sample(ped_ring, num_neurons)

    for neuron in neurons:
        ectoderm.set_stim_pattern("fast", xmin=neuron, xmax=neuron+1, ymin=0, ymax=1, stim_times=stims_fast)

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
    filemeta += "num_neurons=" + str(num_neurons) + ","
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
    parser.add_argument('--num_neurons', type=int, default=5)
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
              args.pathway, args.save_interval, args.num_neurons, args.save_dir,
              T=args.T, dt=args.dt, k_ipr=args.k_ipr, s0=args.s0,
              d=args.d, v_delta=args.v_delta, k_deg=args.k_deg)