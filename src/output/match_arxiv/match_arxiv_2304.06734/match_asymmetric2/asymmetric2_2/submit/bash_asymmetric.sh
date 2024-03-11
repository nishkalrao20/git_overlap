#!/usr/bin/env bash

# asymmetric_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par0 asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation asymmetric2_2/asymmetric_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/asymmetric2_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_V1_2.gwf} --local --label asymmetric_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS asymmetric_data0_1126259462-4702857_generation
# CHILDREN asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis asymmetric2_2/asymmetric_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/asymmetric2_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_V1_2.gwf} --local --outdir asymmetric2_2 --detectors H1 --detectors L1 --detectors V1 --label asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file asymmetric2_2/data/asymmetric_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS asymmetric_data0_1126259462-4702857_generation
# CHILDREN asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis asymmetric2_2/asymmetric_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/asymmetric2_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2304.06734/match_asymmetric2/injections/PAIRS_V1_2.gwf} --local --outdir asymmetric2_2 --detectors H1 --detectors L1 --detectors V1 --label asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file asymmetric2_2/data/asymmetric_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par0 asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_plot asymmetric_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result asymmetric2_2/result/asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 asymmetric2_2/result/asymmetric_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir asymmetric2_2/result --label asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result asymmetric2_2/result/asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir asymmetric2_2/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result asymmetric2_2/result/asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir asymmetric2_2/result --calibration --corner --marginal --skymap --waveform --format png

# asymmetric_pesummary
# PARENTS asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir asymmetric2_2/results_page --config asymmetric2_2/asymmetric_config_complete.ini --samples asymmetric2_2/result/asymmetric_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata asymmetric2_2/data/asymmetric_data0_1126259462-4702857_generation_data_dump.pickle

