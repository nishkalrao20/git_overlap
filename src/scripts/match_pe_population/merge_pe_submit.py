import numpy as np
import itertools
from tqdm import tqdm

mchirpratio, snrratio = [.5, 1., 2.], [.5, 1., 2.]
delta_tc = [-1., -.5, -.3, -.2, -.1, .1, .2, .3, .5, 1.]

combinations = list(itertools.product(mchirpratio, snrratio, delta_tc))

for i in tqdm(range(len(combinations))): 

    with open('/home/nishkal.rao/git_overlap/src/scripts/pe_population/merge_pe.sh', 'a') as bash_file:
        bash_file.write('\ncd /home/nishkal.rao/git_overlap/src/output/pe_population/pe/PE_{}_{}_{}/result/ \npython ~/git_overlap/src/scripts/pe_population/merge_pe.py\n'.format(combinations[i][0], combinations[i][1], combinations[i][2]))