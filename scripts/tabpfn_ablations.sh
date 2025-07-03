#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH --gpus 1
#SBATCH -o logs/%x.%N.%A.%a.out
#SBATCH -e logs/%x.%N.%A.%a.errors
#SBATCH -J T4T-TFN
#SBATCH --mail-type=END,FAIL
#SBATCH -a 1-3 

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with task ID $SLURM_ARRAY_TASK_ID";

export PYTHONUNBUFFERED=1

DATASETS=('it_salary' 'mercari' 'sf_permits')

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID-1]}

echo "Running main_pipeline with dataset $DATASET..."
python -u -m pipelines.main_pipeline --dataset "$DATASET" \
       --project_root /work/dlclarge2/dasb-Camvid/tabadap \
       --run_pipe \
       --eval_method 'tabpfn' \
       --no_text

echo "DONE with dataset $DATASET";
echo "Finished at $(date)";