#!/usr/bin/env bash

# pairs_data0_1126259462-4116447_generation
# PARENTS 
# CHILDREN pairs_data0_1126259462-4116447_analysis_H1L1V1_par0 pairs_data0_1126259462-4116447_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation pairs/pairs_config_complete.ini --submit --label pairs_data0_1126259462-4116447_generation --idx 0 --trigger-time 1126259462.4116447

# pairs_data0_1126259462-4116447_analysis_H1L1V1_par0
# PARENTS pairs_data0_1126259462-4116447_generation
# CHILDREN pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pairs/pairs_config_complete.ini --submit --outdir pairs --detectors H1 --detectors L1 --detectors V1 --label pairs_data0_1126259462-4116447_analysis_H1L1V1_par0 --data-dump-file pairs/data/pairs_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# pairs_data0_1126259462-4116447_analysis_H1L1V1_par1
# PARENTS pairs_data0_1126259462-4116447_generation
# CHILDREN pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis pairs/pairs_config_complete.ini --submit --outdir pairs --detectors H1 --detectors L1 --detectors V1 --label pairs_data0_1126259462-4116447_analysis_H1L1V1_par1 --data-dump-file pairs/data/pairs_data0_1126259462-4116447_generation_data_dump.pickle --sampler dynesty

# pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
# PARENTS pairs_data0_1126259462-4116447_analysis_H1L1V1_par0 pairs_data0_1126259462-4116447_analysis_H1L1V1_par1
# CHILDREN pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_plot pairs_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pairs/result/pairs_data0_1126259462-4116447_analysis_H1L1V1_par0_result.hdf5 pairs/result/pairs_data0_1126259462-4116447_analysis_H1L1V1_par1_result.hdf5 --outdir pairs/result --label pairs_data0_1126259462-4116447_analysis_H1L1V1_merge --extension hdf5 --merge

# pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_final_result
# PARENTS pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result pairs/result/pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir pairs/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_plot
# PARENTS pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result pairs/result/pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 --outdir pairs/result --calibration --corner --marginal --skymap --waveform --format png

# pairs_pesummary
# PARENTS pairs_data0_1126259462-4116447_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir pairs/results_page --config pairs/pairs_config_complete.ini --samples pairs/result/pairs_data0_1126259462-4116447_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata pairs/data/pairs_data0_1126259462-4116447_generation_data_dump.pickle

