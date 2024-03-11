#!/usr/bin/env bash

# pairs_ml_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par0 pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation pe_ml/pairs_ml_config_complete.ini --submit --label pairs_ml_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS pairs_ml_data0_1126259462-4702857_generation
# CHILDREN pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pe_ml/pairs_ml_config_complete.ini --submit --outdir pe_ml --detectors H1 --detectors L1 --detectors V1 --label pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file pe_ml/data/pairs_ml_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS pairs_ml_data0_1126259462-4702857_generation
# CHILDREN pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pe_ml/pairs_ml_config_complete.ini --submit --outdir pe_ml --detectors H1 --detectors L1 --detectors V1 --label pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file pe_ml/data/pairs_ml_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par0 pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_plot pairs_ml_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pe_ml/result/pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 pe_ml/result/pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir pe_ml/result --label pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pe_ml/result/pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir pe_ml/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result pe_ml/result/pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir pe_ml/result --calibration --corner --marginal --skymap --waveform --format png

# pairs_ml_pesummary
# PARENTS pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir pe_ml/results_page --config pe_ml/pairs_ml_config_complete.ini --samples pe_ml/result/pairs_ml_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomPv2 --gwdata pe_ml/data/pairs_ml_data0_1126259462-4702857_generation_data_dump.pickle

