#!/usr/bin/env bash

# singles_a_data0_1126259462-4116447_generation
# PARENTS 
# CHILDREN singles_a_data0_1126259462-4116447_analysis_H1L1V1_par0 singles_a_data0_1126259462-4116447_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation singles_a/singles_a_config_complete.ini --submit --label singles_a_data0_1126259462-4116447_generation --idx 0 --trigger-time 1126259462.4116447

# singles_a_data0_1126259462-4116447_analysis_H1L1V1_par0
# PARENTS singles_a_data0_1126259462-4116447_generation
# CHILDREN singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis singles_a/singles_a_config_complete.ini --submit --outdir singles_a --detectors H1 --detectors L1 --detectors V1 --label singles_a_data0_1126259462-4116447_analysis_H1L1V1_par0 --data-dump-file singles_a/data/singles_a_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# singles_a_data0_1126259462-4116447_analysis_H1L1V1_par1
# PARENTS singles_a_data0_1126259462-4116447_generation
# CHILDREN singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis singles_a/singles_a_config_complete.ini --submit --outdir singles_a --detectors H1 --detectors L1 --detectors V1 --label singles_a_data0_1126259462-4116447_analysis_H1L1V1_par1 --data-dump-file singles_a/data/singles_a_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
# PARENTS singles_a_data0_1126259462-4116447_analysis_H1L1V1_par0 singles_a_data0_1126259462-4116447_analysis_H1L1V1_par1
# CHILDREN singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_plot singles_a_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result singles_a/result/singles_a_data0_1126259462-4116447_analysis_H1L1V1_par0_result.hdf5 singles_a/result/singles_a_data0_1126259462-4116447_analysis_H1L1V1_par1_result.hdf5 --outdir singles_a/result --label singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge --extension hdf5 --merge

# singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result
# PARENTS singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result singles_a/result/singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir singles_a/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_plot
# PARENTS singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result singles_a/result/singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir singles_a/result --calibration --corner --marginal --skymap --waveform --format png

# singles_a_pesummary
# PARENTS singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir singles_a/results_page --config singles_a/singles_a_config_complete.ini --samples singles_a/result/singles_a_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata singles_a/data/singles_a_data0_1126259462-4116447_generation_data_dump.pickle

