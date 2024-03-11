#!/usr/bin/env bash

# PE_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN PE_data0_1126259462-4702857_analysis_H1L1V1_par0 PE_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_pipe_generation outdir/PE_config_complete.ini --submit --label PE_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# PE_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS PE_data0_1126259462-4702857_generation
# CHILDREN PE_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_pipe_analysis outdir/PE_config_complete.ini --submit --outdir outdir --detectors H1 --detectors L1 --detectors V1 --label PE_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file outdir/data/PE_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# PE_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS PE_data0_1126259462-4702857_generation
# CHILDREN PE_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_pipe_analysis outdir/PE_config_complete.ini --submit --outdir outdir --detectors H1 --detectors L1 --detectors V1 --label PE_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file outdir/data/PE_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# PE_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS PE_data0_1126259462-4702857_analysis_H1L1V1_par0 PE_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN PE_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result PE_data0_1126259462-4702857_analysis_H1L1V1_merge_plot PE_pesummary
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_result --result outdir/result/PE_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 outdir/result/PE_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir outdir/result --label PE_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# PE_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS PE_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_result --result outdir/result/PE_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir outdir/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# PE_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS PE_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/bilby_pipe_plot --result outdir/result/PE_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir outdir/result --calibration --corner --marginal --skymap --waveform --format png

# PE_pesummary
# PARENTS PE_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/gw_analysis/bin/summarypages --webdir outdir/results_page --config outdir/PE_config_complete.ini --samples outdir/result/PE_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata outdir/data/PE_data0_1126259462-4702857_generation_data_dump.pickle

