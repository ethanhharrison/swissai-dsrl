#!/bin/bash
# Submit multi-task DSRL runs on libero_90 tasks {28, 29, 30, 31, 32}.
#
# Submit:  bash sbatch_multitask_28-32_all.sh
set -euo pipefail

REPO=/global/home/users/ehharrison/swissai-dsrl
LOGDIR=$REPO/slurm_logs
mkdir -p "$LOGDIR"

CONDA_SETUP='source /global/home/users/ehharrison/miniconda3/etc/profile.d/conda.sh && conda activate arli'
RUN_CMD="cd ${REPO} && DSRL_MULTI_TASK_IDS=28,29,30,31,32 bash ${REPO}/examples/scripts/run_libero_multitask.sh 20"

jobid0=$(sbatch -A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n 1 -c 4 --qos=rail_gpu4_high -t 72:00:00 --mem=60G \
  -J dsrl_multi28-32_utd20_s0 \
  -o "${LOGDIR}/dsrl_multi28-32_utd20_s0.%j.out" \
  -e "${LOGDIR}/dsrl_multi28-32_utd20_s0.%j.err" \
  --requeue --parsable \
  --wrap "${CONDA_SETUP} && ${RUN_CMD} 0 --wandb_project DSRL_pi0_Libero_MultiTask") \
  && echo "submitted $jobid0  (dsrl_multi28-32_utd20_s0)"

jobid1=$(sbatch -A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n 1 -c 4 --qos=rail_gpu4_high -t 72:00:00 --mem=60G \
  -J dsrl_multi28-32_utd20_s1 \
  -o "${LOGDIR}/dsrl_multi28-32_utd20_s1.%j.out" \
  -e "${LOGDIR}/dsrl_multi28-32_utd20_s1.%j.err" \
  --requeue --parsable \
  --wrap "${CONDA_SETUP} && ${RUN_CMD} 1 --wandb_project DSRL_pi0_Libero_MultiTask") \
  && echo "submitted $jobid1  (dsrl_multi28-32_utd20_s1)"
