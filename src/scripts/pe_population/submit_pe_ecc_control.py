import os
import itertools
import numpy as np
from tqdm import tqdm
from glob import glob
from subprocess import call

mchirpratio, snrratio = [.5, 1., 2.], [.5, 1.]
delta_tc = [-.1, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .1]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

dir = os.getcwd()
os.chdir(dir)

for i in tqdm(range(len(combinations))): 

    outdir = dir+'/pe/PE_ECC_CONTROL_{}_{}_{}/'.format(combinations[i][0], combinations[i][1], combinations[i][2])
    
    directory = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/'
    data_H1 = directory+'PAIRS_H1_{}_{}_{}.gwf'.format(combinations[i][0], combinations[i][1], combinations[i][2])
    data_L1 = directory+'PAIRS_L1_{}_{}_{}.gwf'.format(combinations[i][0], combinations[i][1], combinations[i][2])
    data_V1 = directory+'PAIRS_V1_{}_{}_{}.gwf'.format(combinations[i][0], combinations[i][1], combinations[i][2])
    
    try:
        os.mkdir(outdir)
    except OSError as e:
        print(e)

    ini_file = '/home/nishkal.rao/git_overlap/src/scripts/pe_population/config_ecc_control.ini'
    call("cp %s %s"%(ini_file, outdir), shell=1)
    os.chdir(outdir)

    with open('config_ecc_control.ini', 'r') as base_file:
        ini_template = base_file.read() 
    ini_content = ini_template.format(outdir='outdir', data_dict_H1=data_H1, data_dict_L1=data_L1, data_dict_V1=data_V1) 
    with open('config_ecc_control.ini', 'w') as ini_file: 
        ini_file.write(ini_content)

    call('bilby_pipe config_ecc_control.ini --submit', shell=1)