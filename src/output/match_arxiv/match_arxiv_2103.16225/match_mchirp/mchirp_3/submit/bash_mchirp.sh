#!/usr/bin/env bash

# mchirp_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN mchirp_data0_1126259462-4702857_analysis_H1L1V1_par0 mchirp_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation mchirp_3/mchirp_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/mchirp_3 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_H1_3.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_L1_3.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_V1_3.gwf} --local --label mchirp_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# mchirp_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS mchirp_data0_1126259462-4702857_generation
# CHILDREN mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis mchirp_3/mchirp_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/mchirp_3 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_H1_3.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_L1_3.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_V1_3.gwf} --local --outdir mchirp_3 --detectors H1 --detectors L1 --detectors V1 --label mchirp_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file mchirp_3/data/mchirp_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# mchirp_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS mchirp_data0_1126259462-4702857_generation
# CHILDREN mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis mchirp_3/mchirp_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/mchirp_3 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_H1_3.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_L1_3.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_mchirp/injections/PAIRS_V1_3.gwf} --local --outdir mchirp_3 --detectors H1 --detectors L1 --detectors V1 --label mchirp_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file mchirp_3/data/mchirp_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS mchirp_data0_1126259462-4702857_analysis_H1L1V1_par0 mchirp_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_plot mchirp_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result mchirp_3/result/mchirp_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 mchirp_3/result/mchirp_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir mchirp_3/result --label mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result mchirp_3/result/mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir mchirp_3/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result mchirp_3/result/mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir mchirp_3/result --calibration --corner --marginal --skymap --waveform --format png

# mchirp_pesummary
# PARENTS mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir mchirp_3/results_page --config mchirp_3/mchirp_config_complete.ini --samples mchirp_3/result/mchirp_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata mchirp_3/data/mchirp_data0_1126259462-4702857_generation_data_dump.pickle

