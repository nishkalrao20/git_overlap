#!/usr/bin/env bash

# pairs_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN pairs_data0_1126259462-4702857_analysis_H1L1V1_par0 pairs_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation pe/pairs_config_complete.ini --submit --label pairs_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# pairs_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS pairs_data0_1126259462-4702857_generation
# CHILDREN pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pe/pairs_config_complete.ini --submit --outdir pe --detectors H1 --detectors L1 --detectors V1 --label pairs_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file pe/data/pairs_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# pairs_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS pairs_data0_1126259462-4702857_generation
# CHILDREN pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pe/pairs_config_complete.ini --submit --outdir pe --detectors H1 --detectors L1 --detectors V1 --label pairs_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file pe/data/pairs_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS pairs_data0_1126259462-4702857_analysis_H1L1V1_par0 pairs_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_plot pairs_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pe/result/pairs_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 pe/result/pairs_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir pe/result --label pairs_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pe/result/pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir pe/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result pe/result/pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir pe/result --calibration --corner --marginal --skymap --waveform --format png

# pairs_pesummary
# PARENTS pairs_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir pe/results_page --config pe/pairs_config_complete.ini --samples pe/result/pairs_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata pe/data/pairs_data0_1126259462-4702857_generation_data_dump.pickle

