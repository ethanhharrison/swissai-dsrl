#!/bin/bash
# Submit LIBERO inference-delay sweep (pi05_libero, action_horizon=10).
#
# Grid:
#   tasks: 2, 14, 28, 38, 59, 60
#   execution horizon (query_freq): 5
#   inference_delay: 1, 2, 3
#   condition_on_prev_actions: 0 (default; set to 1 to enable, uses num_prev=inference_delay)
#   rl_inference_delay: 0 (default; add values < inference_delay to sweep d')
#   seeds: 0, 1
#
# Optional env overrides:
#   DSRL_CONDITION_ON_PREV_LIST     Comma-separated 0/1, e.g. "0,1"
#   DSRL_RL_INFERENCE_DELAY_LIST    Comma-separated delays, e.g. "0,1,2"
#
# Total jobs: n_tasks * n_delays * n_rl_delays * n_seeds * n_condition_on_prev
# (pairs with rl_delay >= inference_delay are skipped)
#
# Submit from repo root:
#   bash examples/scripts/sbatch_inference_delay_libero.sh
set -euo pipefail

REPO=/global/home/users/ehharrison/dsrl_pi0
mkdir -p "$REPO/slurm_logs"

TASKS=(2 14 28 38 59 60)
DELAYS=(1 2 3)
SEEDS=(0 1)
# Prev-action conditioning: 1 sets --num_prev_actions = inference_delay for that job.
CONDITION_ON_PREV=(0)
if [ -n "${DSRL_CONDITION_ON_PREV_LIST:-}" ]; then
    IFS=',' read -ra CONDITION_ON_PREV <<< "${DSRL_CONDITION_ON_PREV_LIST}"
fi
RL_INFERENCE_DELAYS=(0)
if [ -n "${DSRL_RL_INFERENCE_DELAY_LIST:-}" ]; then
    IFS=',' read -ra RL_INFERENCE_DELAYS <<< "${DSRL_RL_INFERENCE_DELAY_LIST}"
fi
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
        for rl_delay in "${RL_INFERENCE_DELAYS[@]}"; do
            if [ "$rl_delay" -gt 0 ] && [ "$rl_delay" -ge "$delay" ]; then
                echo "skip rl_inference_delay=${rl_delay} (must be < inference_delay=${delay})"
                continue
            fi
            for cond_prev in "${CONDITION_ON_PREV[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    job_name="dsrl_t${task}_qf${QUERY_FREQ}_id${delay}_s${seed}"
                    if [ "$rl_delay" -gt 0 ]; then
                        job_name="${job_name}_rld${rl_delay}"
                    fi
                    if [ "$cond_prev" -eq 1 ]; then
                        job_name="${job_name}_pca"
                    fi
                    log_base="$REPO/slurm_logs/${job_name}"
                    wrap_cmd="${CONDA_SETUP} && cd ${REPO} && DSRL_WANDB_PROJECT=${WANDB_PROJECT} DSRL_CONDITION_ON_PREV_ACTIONS=${cond_prev} DSRL_RL_INFERENCE_DELAY=${rl_delay} bash ${REPO}/examples/scripts/run_inference_delay.sh libero ${QUERY_FREQ} ${delay} ${MULTI_GRAD_STEP} ${seed} ${task} --max_online_trajs ${MAX_ONLINE_TRAJS}"

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
    done
done

echo "Done: submitted ${job_idx} LIBERO inference-delay jobs."
