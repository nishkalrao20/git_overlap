#!/bin/bash

# Define the parameters
mchirpratio=(0.5 1.0 2.0)
snrratio=(0.5 1.0)
delta_tc=(-0.1 -0.05 -0.03 -0.02 -0.01 0.01 0.02 0.03 0.05 0.1)

# Remote server details
remote_user="nishkal.rao"
remote_host="ldas-pcdev6.gw.iucaa.in"
remote_base_path="~/git_overlap/src/output/pe_population/pe"

# Local destination base directory
local_base_path="git_overlap/src/output/pe_population/final_result"

# Loop over all combinations of mchirpratio, snrratio, and delta_tc
for mc in "${mchirpratio[@]}"; do
    for snr in "${snrratio[@]}"; do
        for dtc in "${delta_tc[@]}"; do
            folder_name="PE_ECC_${mc}_${snr}_${dtc}"
            echo "Copying files for $folder_name..."
            
            # Construct the full remote path
            remote_path="${remote_base_path}/${folder_name}/outdir/final_result/*"
            
            # Construct the local destination folder and create it if it doesn't exist
            local_folder="${local_base_path}/${folder_name}"
            mkdir -p "$local_folder"
            
            # Copy the files using scp
            scp -r "${remote_user}@${remote_host}:${remote_path}" "$local_folder"
            
            echo "Finished copying files for $folder_name to $local_folder"
        done
    done
done

echo "All files have been copied successfully."

# # Define the parameters
# mchirpratio=(0.5 1.0 2.0)
# snrratio=(0.5 1.0)
# delta_tc=(-0.1 -0.05 -0.03 -0.02 -0.01 0.01 0.02 0.03 0.05 0.1)

# # Remote server details
# remote_user="nishkal.rao"
# remote_host="ldas-pcdev4.gw.iucaa.in"
# remote_base_path="/home/anuj.mishra/git_repos/git_overlap/pe_runs/pe_files/outdir_mlpe"

# # Local destination base directory
# local_base_path="git_overlap/src/output/pe_population/final_result"

# # Loop over all combinations of mchirpratio, snrratio, and delta_tc
# i=0
# for mc in "${mchirpratio[@]}"; do
#     for snr in "${snrratio[@]}"; do
#         for dtc in "${delta_tc[@]}"; do
#             local_folder_name="PE_ML_${mc}_${snr}_${dtc}"
#             folder_name="PE_ML_Inj_${i}_${mc}_${snr}_${dtc}"
#             file_name="outdir/final_result/PE_ML_Inj_${i}_data0_1126259462-4702857_analysis_H1L1V1_merge_result.hdf5"
#             echo "Copying files for $folder_name..."
            
#             # Construct the full remote path
#             remote_path="${remote_base_path}/${folder_name}/${file_name}"
            
#             # Construct the local destination folder and create it if it doesn't exist
#             local_folder="${local_base_path}/${local_folder_name}"
#             mkdir -p "$local_folder"
            
#             # Copy the files using scp
#             scp -r "${remote_user}@${remote_host}:${remote_path}" "$local_folder"
            
#             echo "Finished copying files for $folder_name to $local_folder"
#             i=$((i+1))
#         done
#     done
# done

# echo "All files have been copied successfully."