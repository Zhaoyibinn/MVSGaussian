#!/bin/bash

__conda_setup="$('/home/zhaoyibin/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/zhaoyibin/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/zhaoyibin/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/zhaoyibin/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
cd FOLDER_PATH="/home/zhaoyibin/3DRE/3DGS/MVSGaussian"
conda activate mvsgs
FOLDER_PATH="/home/zhaoyibin/3DRE/3DGS/MVSGaussian/dtu_data"
for file in "$FOLDER_PATH"/*
do
    echo "当前训练的文件夹是$file"
    python lib/colmap/imgs2poses.py -s "$file"
    # python train.py --source_path "$file" --beta 5.0 --lambda_pearson 0.05 --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 --SDS_freq 0.1 --step_ratio 0.99 --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 --iterations 30000 --checkpoint_iterations 30000 -r 2
done