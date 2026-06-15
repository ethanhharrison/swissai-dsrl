#!/bin/bash
# Submit ALOHA residual-policy sweep via run_residual_aloha_config.sh.
#
# Default grid (override arrays or *_LIST env vars below):
#   query_freq: 25
#   residual_edit_mode: chunk
#   action_magnitude: 0.05
#   inference_delay: 0
#   rl_inference_delay: 0
#   condition_on_prev_actions: 0
#   temp_mode: fixed
#   seeds: 2, 3
#
# Optional env overrides:
#   DSRL_REPO                       Repo root (default: auto-detect from script path)
#   DSRL_CONDA_SETUP                Conda activation command
#   DSRL_RESIDUAL_EDIT_MODE_LIST    Comma-separated step/chunk, e.g. "chunk,step"
#   DSRL_ACTION_MAGNITUDE_LIST      Comma-separated floats, e.g. "0.05,0.1"
#   DSRL_INFERENCE_DELAY_LIST       Comma-separated delays, e.g. "0,10,20"
#   DSRL_RL_INFERENCE_DELAY_LIST    Comma-separated rl delays, e.g. "0,5"
#   DSRL_CONDITION_ON_PREV_LIST     Comma-separated 0/1 (sets num_prev=inference_delay)
#   DSRL_TEMP_MODE_LIST             Comma-separated fixed/learned
#   DSRL_INIT_TEMPERATURE           Passed when temp_mode=fixed (default from config script)
#   DSRL_TARGET_ENTROPY             Passed when temp_mode=learned (default: auto)
#   DSRL_MULTI_GRAD_STEP            UTD (default: 1)
#   DSRL_WANDB_PROJECT              W&B project name
#   DSRL_MAX_ONLINE_TRAJS           Extra arg to training launch
#   DSRL_JOBS_PER_GPU               Pack N ablations per slurm GPU job (default: 1)
#
# Pairs with rl_delay >= inference_delay are skipped.
#
# Submit from repo root:
#   bash examples/scripts/sbatch_residual_aloha.sh
#   DSRL_JOBS_PER_GPU=2 bash examples/scripts/sbatch_residual_aloha.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${DSRL_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
mkdir -p "$REPO/slurm_logs"

QUERY_FREQ=${DSRL_QUERY_FREQ:-25}
MULTI_GRAD_STEP=${DSRL_MULTI_GRAD_STEP:-25}
WANDB_PROJECT=${DSRL_WANDB_PROJECT:-DSRL_pi0_Residual_Aloha}
MAX_ONLINE_TRAJS=${DSRL_MAX_ONLINE_TRAJS:-0}
CONDA_SETUP="${DSRL_CONDA_SETUP:-source /global/home/users/ehharrison/miniconda3/etc/profile.d/conda.sh && conda activate dsrl_pi0}"

RESIDUAL_EDIT_MODES=(chunk)
if [ -n "${DSRL_RESIDUAL_EDIT_MODE_LIST:-}" ]; then
    IFS=',' read -ra RESIDUAL_EDIT_MODES <<< "${DSRL_RESIDUAL_EDIT_MODE_LIST}"
fi

ACTION_MAGNITUDES=(0.01 0.05 0.10)
if [ -n "${DSRL_ACTION_MAGNITUDE_LIST:-}" ]; then
    IFS=',' read -ra ACTION_MAGNITUDES <<< "${DSRL_ACTION_MAGNITUDE_LIST}"
fi

DELAYS=(0)
if [ -n "${DSRL_INFERENCE_DELAY_LIST:-}" ]; then
    IFS=',' read -ra DELAYS <<< "${DSRL_INFERENCE_DELAY_LIST}"
fi

RL_INFERENCE_DELAYS=(0)
if [ -n "${DSRL_RL_INFERENCE_DELAY_LIST:-}" ]; then
    IFS=',' read -ra RL_INFERENCE_DELAYS <<< "${DSRL_RL_INFERENCE_DELAY_LIST}"
fi

CONDITION_ON_PREV=(0)
if [ -n "${DSRL_CONDITION_ON_PREV_LIST:-}" ]; then
    IFS=',' read -ra CONDITION_ON_PREV <<< "${DSRL_CONDITION_ON_PREV_LIST}"
fi

TEMP_MODES=(fixed)
if [ -n "${DSRL_TEMP_MODE_LIST:-}" ]; then
    IFS=',' read -ra TEMP_MODES <<< "${DSRL_TEMP_MODE_LIST}"
fi

SEEDS=(0 1)
if [ -n "${DSRL_SEED_LIST:-}" ]; then
    IFS=',' read -ra SEEDS <<< "${DSRL_SEED_LIST}"
fi

INIT_TEMPERATURE="${DSRL_INIT_TEMPERATURE:-}"
TARGET_ENTROPY="${DSRL_TARGET_ENTROPY:-auto}"

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

for edit_mode in "${RESIDUAL_EDIT_MODES[@]}"; do
    for action_mag in "${ACTION_MAGNITUDES[@]}"; do
        for delay in "${DELAYS[@]}"; do
            for rl_delay in "${RL_INFERENCE_DELAYS[@]}"; do
                if [ "$rl_delay" -gt 0 ] && [ "$rl_delay" -ge "$delay" ]; then
                    echo "skip rl_inference_delay=${rl_delay} (must be < inference_delay=${delay})"
                    continue
                fi
                for cond_prev in "${CONDITION_ON_PREV[@]}"; do
                    if [ "$cond_prev" -eq 1 ] && [ "$delay" -le 0 ]; then
                        echo "skip condition_on_prev=1 with inference_delay=${delay}"
                        continue
                    fi
                    for temp_mode in "${TEMP_MODES[@]}"; do
                        for seed in "${SEEDS[@]}"; do
                            job_name="res_aloha_qf${QUERY_FREQ}_${edit_mode}_am${action_mag}_id${delay}_s${seed}"
                            if [ "$rl_delay" -gt 0 ]; then
                                job_name="${job_name}_rld${rl_delay}"
                            fi
                            if [ "$cond_prev" -eq 1 ]; then
                                job_name="${job_name}_pca"
                            fi
                            if [ "$temp_mode" = "fixed" ]; then
                                if [ -n "$INIT_TEMPERATURE" ]; then
                                    job_name="${job_name}_fixt${INIT_TEMPERATURE}"
                                else
                                    job_name="${job_name}_fixt"
                                fi
                            else
                                job_name="${job_name}_lt${TARGET_ENTROPY}"
                            fi

                            log_base="$REPO/slurm_logs/${job_name}"

                            env_prefix="DSRL_WANDB_PROJECT=${WANDB_PROJECT}"
                            env_prefix="${env_prefix} DSRL_RESIDUAL_EDIT_MODE=${edit_mode}"
                            env_prefix="${env_prefix} DSRL_TEMP_MODE=${temp_mode}"
                            env_prefix="${env_prefix} DSRL_CONDITION_ON_PREV_ACTIONS=${cond_prev}"
                            if [ -n "$INIT_TEMPERATURE" ]; then
                                env_prefix="${env_prefix} DSRL_INIT_TEMPERATURE=${INIT_TEMPERATURE}"
                            fi
                            if [ "$temp_mode" = "learned" ]; then
                                env_prefix="${env_prefix} DSRL_TARGET_ENTROPY=${TARGET_ENTROPY}"
                            fi

                            extra_train_args=()
                            if [ "$MAX_ONLINE_TRAJS" -gt 0 ]; then
                                extra_train_args=(--max_online_trajs "$MAX_ONLINE_TRAJS")
                            fi

                            wrap_cmd="${CONDA_SETUP} && cd ${REPO} && ${env_prefix} bash ${REPO}/examples/scripts/run_residual_aloha_config.sh ${QUERY_FREQ} ${MULTI_GRAD_STEP} ${seed} ${action_mag} ${delay} ${rl_delay} 0 ${extra_train_args[*]}"

                            WRAP_CMDS+=("$wrap_cmd")
                            JOB_NAMES+=("$job_name")
                            LOG_BASES+=("$log_base")
                        done
                    done
                done
            done
        done
    done
done

# shellcheck source=examples/scripts/sbatch_inference_delay_submit.sh
source "${SCRIPT_DIR}/sbatch_inference_delay_submit.sh"
submit_inference_delay_sweep
