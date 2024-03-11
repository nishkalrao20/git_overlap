#!/usr/bin/env bash

# singles_b_data0_1126259462-6616447_generation
# PARENTS 
# CHILDREN singles_b_data0_1126259462-6616447_analysis_H1L1V1_par0 singles_b_data0_1126259462-6616447_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation singles_b/singles_b_config_complete.ini --submit --label singles_b_data0_1126259462-6616447_generation --idx 0 --trigger-time 1126259462.6616447

# singles_b_data0_1126259462-6616447_analysis_H1L1V1_par0
# PARENTS singles_b_data0_1126259462-6616447_generation
# CHILDREN singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis singles_b/singles_b_config_complete.ini --submit --outdir singles_b --detectors H1 --detectors L1 --detectors V1 --label singles_b_data0_1126259462-6616447_analysis_H1L1V1_par0 --data-dump-file singles_b/data/singles_b_data0_1126259462-6616447_generation_data_dump.pickle --sampler dynesty

# singles_b_data0_1126259462-6616447_analysis_H1L1V1_par1
# PARENTS singles_b_data0_1126259462-6616447_generation
# CHILDREN singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis singles_b/singles_b_config_complete.ini --submit --outdir singles_b --detectors H1 --detectors L1 --detectors V1 --label singles_b_data0_1126259462-6616447_analysis_H1L1V1_par1 --data-dump-file singles_b/data/singles_b_data0_1126259462-6616447_generation_data_dump.pickle --sampler dynesty

# singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
# PARENTS singles_b_data0_1126259462-6616447_analysis_H1L1V1_par0 singles_b_data0_1126259462-6616447_analysis_H1L1V1_par1
# CHILDREN singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_final_result singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_plot singles_b_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result singles_b/result/singles_b_data0_1126259462-6616447_analysis_H1L1V1_par0_result.hdf5 singles_b/result/singles_b_data0_1126259462-6616447_analysis_H1L1V1_par1_result.hdf5 --outdir singles_b/result --label singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge --extension hdf5 --merge

# singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_final_result
# PARENTS singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result singles_b/result/singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_result.hdf5 --outdir singles_b/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_plot
# PARENTS singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result singles_b/result/singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_result.hdf5 --outdir singles_b/result --calibration --corner --marginal --skymap --waveform --format png

# singles_b_pesummary
# PARENTS singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir singles_b/results_page --config singles_b/singles_b_config_complete.ini --samples singles_b/result/singles_b_data0_1126259462-6616447_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata singles_b/data/singles_b_data0_1126259462-6616447_generation_data_dump.pickle

