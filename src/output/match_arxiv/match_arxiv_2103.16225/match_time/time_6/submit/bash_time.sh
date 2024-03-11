#!/usr/bin/env bash

# time_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN time_data0_1126259462-4702857_analysis_H1L1V1_par0 time_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation time_6/time_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/time_6 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_H1_6.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_L1_6.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_V1_6.gwf} --local --label time_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# time_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS time_data0_1126259462-4702857_generation
# CHILDREN time_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis time_6/time_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/time_6 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_H1_6.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_L1_6.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_V1_6.gwf} --local --outdir time_6 --detectors H1 --detectors L1 --detectors V1 --label time_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file time_6/data/time_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# time_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS time_data0_1126259462-4702857_generation
# CHILDREN time_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis time_6/time_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/time_6 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_H1_6.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_L1_6.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time/injections/PAIRS_V1_6.gwf} --local --outdir time_6 --detectors H1 --detectors L1 --detectors V1 --label time_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file time_6/data/time_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# time_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS time_data0_1126259462-4702857_analysis_H1L1V1_par0 time_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN time_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result time_data0_1126259462-4702857_analysis_H1L1V1_merge_plot time_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result time_6/result/time_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 time_6/result/time_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir time_6/result --label time_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# time_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS time_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result time_6/result/time_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir time_6/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# time_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS time_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result time_6/result/time_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir time_6/result --calibration --corner --marginal --skymap --waveform --format png

# time_pesummary
# PARENTS time_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir time_6/results_page --config time_6/time_config_complete.ini --samples time_6/result/time_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata time_6/data/time_data0_1126259462-4702857_generation_data_dump.pickle

