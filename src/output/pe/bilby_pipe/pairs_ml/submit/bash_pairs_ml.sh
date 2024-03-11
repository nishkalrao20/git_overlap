#!/usr/bin/env bash

# pairs_ml_data0_1126259462-4116447_generation
# PARENTS 
# CHILDREN pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par0 pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation pairs_ml/pairs_ml_config_complete.ini --submit --label pairs_ml_data0_1126259462-4116447_generation --idx 0 --trigger-time 1126259462.4116447

# pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par0
# PARENTS pairs_ml_data0_1126259462-4116447_generation
# CHILDREN pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pairs_ml/pairs_ml_config_complete.ini --submit --outdir pairs_ml --detectors H1 --detectors L1 --detectors V1 --label pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par0 --data-dump-file pairs_ml/data/pairs_ml_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par1
# PARENTS pairs_ml_data0_1126259462-4116447_generation
# CHILDREN pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pairs_ml/pairs_ml_config_complete.ini --submit --outdir pairs_ml --detectors H1 --detectors L1 --detectors V1 --label pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par1 --data-dump-file pairs_ml/data/pairs_ml_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
# PARENTS pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par0 pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par1
# CHILDREN pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_plot pairs_ml_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pairs_ml/result/pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par0_result.hdf5 pairs_ml/result/pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_par1_result.hdf5 --outdir pairs_ml/result --label pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge --extension hdf5 --merge

# pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result
# PARENTS pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pairs_ml/result/pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir pairs_ml/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_plot
# PARENTS pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result pairs_ml/result/pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir pairs_ml/result --calibration --corner --marginal --skymap --waveform --format png

# pairs_ml_pesummary
# PARENTS pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir pairs_ml/results_page --config pairs_ml/pairs_ml_config_complete.ini --samples pairs_ml/result/pairs_ml_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomPv2 --gwdata pairs_ml/data/pairs_ml_data0_1126259462-4116447_generation_data_dump.pickle

