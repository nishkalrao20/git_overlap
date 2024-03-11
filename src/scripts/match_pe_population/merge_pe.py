import numpy as np
import bilby
from glob import glob

fs = glob('*par*.hdf5')
res = []
for f in fs:
    print(f)
    try:
        res.append(bilby.result.read_in_result(f))
    except:
        try:
            f=f.split('.hdf5')[0] + '.pkl'
            print(f)
            res.append(bilby.result.read_in_result(f))
        except:
            pass


cres = bilby.core.result.ResultList(res).combine()

cres.save_to_file(res[0].label.split('par')[0]+'merge_result.json', outdir='.')