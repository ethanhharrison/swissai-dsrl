#!/usr/bin/env bash
# Shared batch sbatch submission for inference-delay sweep scripts.
#
# Caller must define bash arrays before sourcing and calling
# submit_inference_delay_sweep:
#   WRAP_CMDS[]   shell command for each ablation
#   JOB_NAMES[]   slurm job name per ablation
#   LOG_BASES[]   log path prefix (no .out suffix)
#   SBATCH_BASE[] sbatch option array
#   REPO          repo root path
#
# Env:
#   DSRL_JOBS_PER_GPU  Ablation runs packed per slurm allocation (default: 1).
submit_inference_delay_sweep() {
    local jobs_per_gpu="${DSRL_JOBS_PER_GPU:-1}"
    if ! [[ "$jobs_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
        echo "DSRL_JOBS_PER_GPU must be a positive integer (got: $jobs_per_gpu)" >&2
        return 1
    fi

    local n=${#WRAP_CMDS[@]}
    if [[ "$n" -ne "${#JOB_NAMES[@]}" ]] || [[ "$n" -ne "${#LOG_BASES[@]}" ]]; then
        echo "WRAP_CMDS, JOB_NAMES, and LOG_BASES must have the same length" >&2
        return 1
    fi
    if [[ "$n" -eq 0 ]]; then
        echo "No jobs to submit."
        return 0
    fi

    local num_slurm_jobs=$(( (n + jobs_per_gpu - 1) / jobs_per_gpu ))
    echo "Packing ${n} ablation(s) into ${num_slurm_jobs} slurm job(s) (${jobs_per_gpu} per GPU)"

    local job_idx=0
    local batch_start=0
    while (( batch_start < n )); do
        local batch_end=$(( batch_start + jobs_per_gpu ))
        if (( batch_end > n )); then
            batch_end=$n
        fi
        local batch_size=$(( batch_end - batch_start ))

        local slurm_job_name="${JOB_NAMES[$batch_start]}"
        if (( batch_size > 1 )); then
            slurm_job_name="${slurm_job_name}_plus$(( batch_size - 1 ))"
        fi

        local wrap_payload=""
        local j
        if (( batch_size == 1 )); then
            wrap_payload="${WRAP_CMDS[$batch_start]}"
        else
            for (( j = batch_start; j < batch_end; j++ )); do
                local log_out="${LOG_BASES[$j]}.\${SLURM_JOB_ID}.out"
                local log_err="${LOG_BASES[$j]}.\${SLURM_JOB_ID}.err"
                local part="(${WRAP_CMDS[$j]}) > ${log_out} 2> ${log_err}"
                if [[ -z "$wrap_payload" ]]; then
                    wrap_payload="$part"
                else
                    wrap_payload="${wrap_payload} & ${part}"
                fi
            done
            wrap_payload="${wrap_payload} & wait"
        fi

        local log_base="${LOG_BASES[$batch_start]}"
        local jobid
        jobid=$(sbatch "${SBATCH_BASE[@]}" \
            -J "$slurm_job_name" \
            -o "${log_base}.%j.out" \
            -e "${log_base}.%j.err" \
            --wrap "$wrap_payload")

        echo -n "submitted slurm ${job_idx}"
        if (( batch_size > 1 )); then
            echo -n "  (${batch_size} ablations:"
            for (( j = batch_start; j < batch_end; j++ )); do
                echo -n " ${JOB_NAMES[$j]}"
            done
            echo -n ")"
        fi
        echo "  ${jobid}  (${slurm_job_name})"

        job_idx=$(( job_idx + 1 ))
        batch_start=$batch_end
    done

    echo "Done: submitted ${job_idx} slurm job(s) for ${n} ablation(s)."
}
