#!/usr/bin/env bash

# snr_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN snr_data0_1126259462-4702857_analysis_H1L1V1_par0 snr_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation snr_2/snr_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/snr_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_V1_2.gwf} --local --label snr_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# snr_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS snr_data0_1126259462-4702857_generation
# CHILDREN snr_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis snr_2/snr_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/snr_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_V1_2.gwf} --local --outdir snr_2 --detectors H1 --detectors L1 --detectors V1 --label snr_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file snr_2/data/snr_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# snr_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS snr_data0_1126259462-4702857_generation
# CHILDREN snr_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis snr_2/snr_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/snr_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_snr/injections/PAIRS_V1_2.gwf} --local --outdir snr_2 --detectors H1 --detectors L1 --detectors V1 --label snr_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file snr_2/data/snr_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# snr_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS snr_data0_1126259462-4702857_analysis_H1L1V1_par0 snr_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN snr_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result snr_data0_1126259462-4702857_analysis_H1L1V1_merge_plot snr_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result snr_2/result/snr_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 snr_2/result/snr_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir snr_2/result --label snr_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# snr_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS snr_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result snr_2/result/snr_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir snr_2/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# snr_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS snr_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result snr_2/result/snr_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir snr_2/result --calibration --corner --marginal --skymap --waveform --format png

# snr_pesummary
# PARENTS snr_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir snr_2/results_page --config snr_2/snr_config_complete.ini --samples snr_2/result/snr_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata snr_2/data/snr_data0_1126259462-4702857_generation_data_dump.pickle

