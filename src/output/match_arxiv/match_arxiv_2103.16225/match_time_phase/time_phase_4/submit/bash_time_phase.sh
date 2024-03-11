#!/usr/bin/env bash

# time_phase_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN time_phase_data0_1126259462-4702857_analysis_H1L1V1_par0 time_phase_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation time_phase_4/time_phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/time_phase_4 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_H1_4.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_L1_4.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_V1_4.gwf} --local --label time_phase_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# time_phase_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS time_phase_data0_1126259462-4702857_generation
# CHILDREN time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis time_phase_4/time_phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/time_phase_4 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_H1_4.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_L1_4.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_V1_4.gwf} --local --outdir time_phase_4 --detectors H1 --detectors L1 --detectors V1 --label time_phase_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file time_phase_4/data/time_phase_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# time_phase_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS time_phase_data0_1126259462-4702857_generation
# CHILDREN time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis time_phase_4/time_phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/time_phase_4 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_H1_4.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_L1_4.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_time_phase/injections/PAIRS_V1_4.gwf} --local --outdir time_phase_4 --detectors H1 --detectors L1 --detectors V1 --label time_phase_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file time_phase_4/data/time_phase_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS time_phase_data0_1126259462-4702857_analysis_H1L1V1_par0 time_phase_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_plot time_phase_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result time_phase_4/result/time_phase_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 time_phase_4/result/time_phase_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir time_phase_4/result --label time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result time_phase_4/result/time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir time_phase_4/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result time_phase_4/result/time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir time_phase_4/result --calibration --corner --marginal --skymap --waveform --format png

# time_phase_pesummary
# PARENTS time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir time_phase_4/results_page --config time_phase_4/time_phase_config_complete.ini --samples time_phase_4/result/time_phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata time_phase_4/data/time_phase_data0_1126259462-4702857_generation_data_dump.pickle

