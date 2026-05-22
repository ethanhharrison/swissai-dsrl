#!/bin/bash
# Submit LIBERO inference-delay sweep (pi05_libero, action_horizon=10).
#
# Grid:
#   tasks: 2, 14, 28, 38, 59, 60
#   execution horizon (query_freq): 5
#   inference_delay: 1, 2, 3
#   seeds: 0, 1
#
# Total jobs: 6 * 3 * 2 = 36
#
# Submit from repo root:
#   bash examples/scripts/sbatch_inference_delay_libero.sh
set -euo pipefail

REPO=/global/home/users/ehharrison/dsrl_pi0
mkdir -p "$REPO/slurm_logs"

TASKS=(2 14 28 38 59 60)
DELAYS=(1 2 3)
SEEDS=(0 1)
QUERY_FREQ=5
MULTI_GRAD_STEP=${DSRL_MULTI_GRAD_STEP:-20}
WANDB_PROJECT=${DSRL_WANDB_PROJECT:-DSRL_pi0_InferenceDelay_Libero}
MAX_ONLINE_TRAJS=${DSRL_MAX_ONLINE_TRAJS:-5000}

CONDA_SETUP='source /global/home/users/ehharrison/miniconda3/etc/profile.d/conda.sh && conda activate dsrl_pi0'
SBATCH_BASE=(
    -A co_rail
    -p savio4_gpu
    --gres=gpu:A5000:1
    -N 1
    -n 1
    -c 4
    --qos=rail_gpu4_high
    -t 72:00:00
    --mem=60G
    --requeue
    --parsable
)

job_idx=0
for task in "${TASKS[@]}"; do
    for delay in "${DELAYS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_name="dsrl_t${task}_qf${QUERY_FREQ}_id${delay}_s${seed}"
            log_base="$REPO/slurm_logs/${job_name}"
            wrap_cmd="${CONDA_SETUP} && cd ${REPO} && DSRL_WANDB_PROJECT=${WANDB_PROJECT} bash ${REPO}/examples/scripts/run_inference_delay.sh libero ${QUERY_FREQ} ${delay} ${MULTI_GRAD_STEP} ${seed} ${task} --max_online_trajs ${MAX_ONLINE_TRAJS}"

            jobid=$(sbatch "${SBATCH_BASE[@]}" \
                -J "$job_name" \
                -o "${log_base}.%j.out" \
                -e "${log_base}.%j.err" \
                --wrap "$wrap_cmd")
            echo "submitted job ${job_idx}  ${jobid}  (${job_name})"
            job_idx=$((job_idx + 1))
        done
    done
done

echo "Done: submitted ${job_idx} LIBERO inference-delay jobs."
