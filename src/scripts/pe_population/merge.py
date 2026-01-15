import os
import itertools
import numpy as np
from tqdm import tqdm
from glob import glob
from subprocess import call
import bilby

mchirpratio, snrratio = [.5, 1., 2.], [.5, 1.]
delta_tc = [-.1, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .1]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

dir = os.getcwd()
os.chdir(dir)

for i in tqdm(range(len(combinations))): 

    outdir = dir+'/pe/PE_ECC_{}_{}_{}/outdir/result/'.format(combinations[i][0], combinations[i][1], combinations[i][2])
    os.chdir(outdir)

    fs = glob('*par*.hdf5')
    print(f"Found {len(fs)} files to merge")
    res = []
    try:
        for f in fs:
            print(f)
            try:
                res.append(bilby.result.read_in_result(f))
            except:
                f=f.split('.hdf5')[0] + '.pkl'
                print(f)
                res.append(bilby.result.read_in_result(f))

        cres = bilby.core.result.ResultList(res).combine()

        merged_file_name = res[0].label.split('par')[0]+'merge_result.pkl'
        cres.save_to_file(merged_file_name, outdir='./', extension="pkl")
        call(f"cp {merged_file_name} ../final_result/", shell=True)
        print(f"Merged file saved as {merged_file_name}")
    except Exception as e:
        print(f"Failed to merge for combination {combinations[i]} with error: {e}")