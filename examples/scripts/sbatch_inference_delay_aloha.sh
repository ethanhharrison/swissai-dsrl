#!/bin/bash
# Submit ALOHA inference-delay sweep (pi0_aloha_sim, action_horizon=50).
#
# Grid:
#   execution horizon (query_freq): 25
#   inference_delay: 2, 5, 10, 20
#   condition_on_prev_actions: 0 (default; set to 1 to enable, uses num_prev=inference_delay)
#   rl_inference_delay: 0 (default; add values < inference_delay to sweep d')
#   seeds: 2, 3
#
# Optional env overrides:
#   DSRL_CONDITION_ON_PREV_LIST     Comma-separated 0/1, e.g. "0,1"
#   DSRL_RL_INFERENCE_DELAY_LIST    Comma-separated delays, e.g. "0,5,10"
#   DSRL_JOBS_PER_GPU               Pack N ablations per slurm GPU job (default: 1).
#                                   Runs execute in parallel (& wait); each ablation
#                                   also writes logs to slurm_logs/<job_name>.<jobid>.out.
#
# Total ablations: n_delays * n_rl_delays * n_seeds * n_condition_on_prev
# (pairs with rl_delay >= inference_delay are skipped)
#
# Submit from repo root:
#   bash examples/scripts/sbatch_inference_delay_aloha.sh
#   DSRL_JOBS_PER_GPU=2 bash examples/scripts/sbatch_inference_delay_aloha.sh
set -euo pipefail

REPO=/global/home/users/ehharrison/dsrl_pi0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$REPO/slurm_logs"

DELAYS=(20)
SEEDS=(3 4 5)
# Prev-action conditioning: 1 sets --num_prev_actions = inference_delay for that job.
CONDITION_ON_PREV=(0 1)
if [ -n "${DSRL_CONDITION_ON_PREV_LIST:-}" ]; then
    IFS=',' read -ra CONDITION_ON_PREV <<< "${DSRL_CONDITION_ON_PREV_LIST}"
fi
# SAC-only fresher observation delay d' (must be 0 or < inference_delay per job).
RL_INFERENCE_DELAYS=(10)
if [ -n "${DSRL_RL_INFERENCE_DELAY_LIST:-}" ]; then
    IFS=',' read -ra RL_INFERENCE_DELAYS <<< "${DSRL_RL_INFERENCE_DELAY_LIST}"
fi
QUERY_FREQ=25
MULTI_GRAD_STEP=${DSRL_MULTI_GRAD_STEP:-25}
WANDB_PROJECT=${DSRL_WANDB_PROJECT:-DSRL_pi0_InferenceDelay_Aloha_Full}
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

WRAP_CMDS=()
JOB_NAMES=()
LOG_BASES=()

for delay in "${DELAYS[@]}"; do
    for rl_delay in "${RL_INFERENCE_DELAYS[@]}"; do
        if [ "$rl_delay" -gt 0 ] && [ "$rl_delay" -ge "$delay" ]; then
            echo "skip rl_inference_delay=${rl_delay} (must be < inference_delay=${delay})"
            continue
        fi
        for cond_prev in "${CONDITION_ON_PREV[@]}"; do
            for seed in "${SEEDS[@]}"; do
                job_name="dsrl_aloha_qf${QUERY_FREQ}_id${delay}_s${seed}"
                if [ "$rl_delay" -gt 0 ]; then
                    job_name="${job_name}_rld${rl_delay}"
                fi
                if [ "$cond_prev" -eq 1 ]; then
                    job_name="${job_name}_pca"
                fi
                log_base="$REPO/slurm_logs/${job_name}"
                wrap_cmd="${CONDA_SETUP} && cd ${REPO} && DSRL_WANDB_PROJECT=${WANDB_PROJECT} DSRL_CONDITION_ON_PREV_ACTIONS=${cond_prev} DSRL_RL_INFERENCE_DELAY=${rl_delay} bash ${REPO}/examples/scripts/run_inference_delay.sh aloha_cube ${QUERY_FREQ} ${delay} ${MULTI_GRAD_STEP} ${seed} --max_online_trajs ${MAX_ONLINE_TRAJS}"

                WRAP_CMDS+=("$wrap_cmd")
                JOB_NAMES+=("$job_name")
                LOG_BASES+=("$log_base")
            done
        done
    done
done

# shellcheck source=examples/scripts/sbatch_inference_delay_submit.sh
source "${SCRIPT_DIR}/sbatch_inference_delay_submit.sh"
submit_inference_delay_sweep
