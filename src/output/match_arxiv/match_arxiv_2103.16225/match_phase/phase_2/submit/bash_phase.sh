#!/usr/bin/env bash

# phase_data0_1126259462-4702857_generation
# PARENTS 
# CHILDREN phase_data0_1126259462-4702857_analysis_H1L1V1_par0 phase_data0_1126259462-4702857_analysis_H1L1V1_par1
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_generation phase_2/phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/phase_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_V1_2.gwf} --local --label phase_data0_1126259462-4702857_generation --idx 0 --trigger-time 1126259462.4702857

# phase_data0_1126259462-4702857_analysis_H1L1V1_par0
# PARENTS phase_data0_1126259462-4702857_generation
# CHILDREN phase_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis phase_2/phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/phase_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_V1_2.gwf} --local --outdir phase_2 --detectors H1 --detectors L1 --detectors V1 --label phase_data0_1126259462-4702857_analysis_H1L1V1_par0 --data-dump-file phase_2/data/phase_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# phase_data0_1126259462-4702857_analysis_H1L1V1_par1
# PARENTS phase_data0_1126259462-4702857_generation
# CHILDREN phase_data0_1126259462-4702857_analysis_H1L1V1_merge
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_analysis phase_2/phase_config_complete.ini --outdir=/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/phase_2 --data-dict={H1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_H1_2.gwf, L1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_L1_2.gwf, V1:/home/nishkal.rao/git_overlap/src/output/match_arxiv/match_arxiv_2103.16225/match_phase/injections/PAIRS_V1_2.gwf} --local --outdir phase_2 --detectors H1 --detectors L1 --detectors V1 --label phase_data0_1126259462-4702857_analysis_H1L1V1_par1 --data-dump-file phase_2/data/phase_data0_1126259462-4702857_generation_data_dump.pickle --sampler dynesty

# phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# PARENTS phase_data0_1126259462-4702857_analysis_H1L1V1_par0 phase_data0_1126259462-4702857_analysis_H1L1V1_par1
# CHILDREN phase_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result phase_data0_1126259462-4702857_analysis_H1L1V1_merge_plot phase_pesummary
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result phase_2/result/phase_data0_1126259462-4702857_analysis_H1L1V1_par0_result.hdf5 phase_2/result/phase_data0_1126259462-4702857_analysis_H1L1V1_par1_result.hdf5 --outdir phase_2/result --label phase_data0_1126259462-4702857_analysis_H1L1V1_merge --extension hdf5 --merge

# phase_data0_1126259462-4702857_analysis_H1L1V1_merge_final_result
# PARENTS phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_result --result phase_2/result/phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir phase_2/final_result --extension hdf5 --max-samples 20000 --lightweight --save

# phase_data0_1126259462-4702857_analysis_H1L1V1_merge_plot
# PARENTS phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/bilby_pipe_plot --result phase_2/result/phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 --outdir phase_2/result --calibration --corner --marginal --skymap --waveform --format png

# phase_pesummary
# PARENTS phase_data0_1126259462-4702857_analysis_H1L1V1_merge
# CHILDREN 
/home/nishkal.rao/miniconda3/envs/igwn_py39-20230425/bin/summarypages --webdir phase_2/results_page --config phase_2/phase_config_complete.ini --samples phase_2/result/phase_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5 -a IMRPhenomXPHM --gwdata phase_2/data/phase_data0_1126259462-4702857_generation_data_dump.pickle

