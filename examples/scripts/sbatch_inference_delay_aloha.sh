#!/bin/bash
# Submit ALOHA inference-delay sweep (pi0_aloha_sim, action_horizon=50).
#
# Grid:
#   execution horizon (query_freq): 25
#   inference_delay: 2, 5, 10, 20
#   condition_on_prev_actions: 0 (default; set to 1 to enable, uses num_prev=inference_delay)
#   seeds: 2, 3
#
# Optional env overrides:
#   DSRL_CONDITION_ON_PREV_LIST  Comma-separated 0/1, e.g. "0,1" (overrides CONDITION_ON_PREV)
#
# Total jobs: n_delays * n_seeds * n_condition_on_prev
#
# Submit from repo root:
#   bash examples/scripts/sbatch_inference_delay_aloha.sh
set -euo pipefail

REPO=/global/home/users/ehharrison/dsrl_pi0
mkdir -p "$REPO/slurm_logs"

DELAYS=(10)
SEEDS=(0)
# Prev-action conditioning: 1 sets --num_prev_actions = inference_delay for that job.
CONDITION_ON_PREV=(1)
if [ -n "${DSRL_CONDITION_ON_PREV_LIST:-}" ]; then
    IFS=',' read -ra CONDITION_ON_PREV <<< "${DSRL_CONDITION_ON_PREV_LIST}"
fi
QUERY_FREQ=25
MULTI_GRAD_STEP=${DSRL_MULTI_GRAD_STEP:-25}
WANDB_PROJECT=${DSRL_WANDB_PROJECT:-DSRL_pi0_InferenceDelay_Aloha}
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
for delay in "${DELAYS[@]}"; do
    for cond_prev in "${CONDITION_ON_PREV[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_name="dsrl_aloha_qf${QUERY_FREQ}_id${delay}_s${seed}"
            if [ "$cond_prev" -eq 1 ]; then
                job_name="${job_name}_pca"
            fi
            log_base="$REPO/slurm_logs/${job_name}"
            wrap_cmd="${CONDA_SETUP} && cd ${REPO} && DSRL_WANDB_PROJECT=${WANDB_PROJECT} DSRL_CONDITION_ON_PREV_ACTIONS=${cond_prev} bash ${REPO}/examples/scripts/run_inference_delay.sh aloha_cube ${QUERY_FREQ} ${delay} ${MULTI_GRAD_STEP} ${seed} --max_online_trajs ${MAX_ONLINE_TRAJS}"

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

echo "Done: submitted ${job_idx} ALOHA inference-delay jobs."
