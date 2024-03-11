import numpy as np
import itertools
from tqdm import tqdm

mchirpratio, snrratio = [.5, 1., 2.], [.5, 1., 2.]
delta_tc = [-1., -.5, -.3, -.2, -.1, .1, .2, .3, .5, 1.]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

with open('/home/nishkal.rao/git_overlap/src/scripts/pe_population/config.ini', 'r') as base_file:
        ini_template = base_file.read() 
 
for i in tqdm(range(len(combinations))): 
    outdir = '/home/nishkal.rao/git_overlap/src/output/pe_population/pe/PE_{}_{}_{}'.format(combinations[i][0], combinations[i][1], combinations[i][2]) 
     
    data_dict_H1 = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format('H1', combinations[i][0], combinations[i][1], combinations[i][2]) 
    data_dict_L1 = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format('L1', combinations[i][0], combinations[i][1], combinations[i][2]) 
    data_dict_V1 = '/home/nishkal.rao/git_overlap/src/output/pe_population/injections/PAIRS_{}_{}_{}_{}.gwf'.format('V1', combinations[i][0], combinations[i][1], combinations[i][2]) 
 
    ini_content = ini_template.format(outdir=outdir, data_dict_H1=data_dict_H1, data_dict_L1=data_dict_L1, data_dict_V1=data_dict_V1) 
     
    with open('/home/nishkal.rao/git_overlap/src/output/pe_population/pe/ini/config_pe_{}_{}_{}.ini'.format(combinations[i][0], combinations[i][1], combinations[i][2]), 'w') as ini_file: 
        ini_file.write(ini_content)

    with open('/home/nishkal.rao/git_overlap/src/scripts/pe_population/submit_pe.sh', 'a') as bash_file:
        bash_file.write('bilby_pipe git_overlap/src/output/pe_population/pe/ini/config_pe_{}_{}_{}.ini --submit\n'.format(combinations[i][0], combinations[i][1], combinations[i][2]))