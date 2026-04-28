"""Generate a BRC (Berkeley Research Computing) sbatch script that submits the
iteration-size x UTD x seed ablation sweep.

Running this file will overwrite ``sbatch_ablation.sh`` in the current working
directory. To actually submit the jobs, run ``bash sbatch_ablation.sh`` after
inspecting it.

Design notes
------------
- One slurm job per (iteration_size, multi_grad_step, seed) combination. The
  pi0 model is large enough that packing >1 python process onto the same A5000
  is unlikely to fit in memory, so we don't use the multi-job-per-GPU trick
  from ``batch_script_example.py``. Set ``NUM_PARALLEL`` > 1 below if you want
  to try it anyway.
- The actual training command lives in
  ``examples/scripts/run_ablation.sh``. That script takes iter_size, utd and
  seed as positional args, sets up all the scratch/cache env vars, and forwards
  everything else to ``python -m examples.launch_train_sim``.
- Wandb run names include iter_size, utd (via ``--prefix`` augmentation in
  ``examples/train_sim.py``) and seed (baked in by ``create_exp_name``), so
  the runs group cleanly for the ablation.

Edit the sweep grid and slurm resources below to taste.
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
ITERATION_SIZES = [100, 200, 500, 1000]
# UTDs (multi_grad_step). Adjust to taste.
MULTI_GRAD_STEPS = [1, 5, 10, 20]
SEEDS = [0, 1]
# LIBERO task id(s) within the libero_90 benchmark to ablate over. Add more
# entries here to sweep tasks too.
TASK_IDS = [2]


# ---------------------------------------------------------------------------
# Slurm / cluster config
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = PROJ_ROOT / "examples" / "scripts" / "run_ablation.sh"

# Name of the wandb project to write runs to.
WANDB_PROJECT = "DSRL_pi0_Libero_Ablation"

# Conda env to activate inside the job. Update to match your setup.
CONDA_ENV = os.environ.get("DSRL_CONDA_ENV", "dsrl_pi0")
# Absolute path to ``conda.sh`` (sourced before activating the env). Change if
# miniconda lives elsewhere on your account.
CONDA_SH = os.environ.get(
    "DSRL_CONDA_SH",
    f"/global/home/users/{os.environ.get('USER', 'ehharrison')}/miniconda3/etc/profile.d/conda.sh",
)

# Slurm options (mirrors batch_script_example.py).
ACCOUNT = "co_rail"
PARTITION = "savio4_gpu"
GRES = "gpu:A5000:1"
PRIORITY = "high"  # -> --qos=rail_gpu4_{priority}
NODES = 1
NTASKS = 1
CPUS_PER_TASK = 4
TIME_LIMIT = "72:00:00"
MEM = "60G"

# Number of python jobs to pack onto one slurm allocation. Keep at 1 for pi0
# on a single A5000; set to >1 only if you know two runs fit.
NUM_PARALLEL = 1

# Extra args forwarded to launch_train_sim (via run_ablation.sh). Override
# anything you want different from run_ablation.sh's defaults here, e.g.
# EXTRA_ARGS = ["--max_steps", "200000"].
EXTRA_ARGS: list[str] = []

OUTPUT_SBATCH = PROJ_ROOT / "sbatch_ablation.sh"


# ---------------------------------------------------------------------------


def build_single_command(iter_size: int, utd: int, seed: int, task_id: int) -> str:
    """Return the shell command that runs one python job."""
    extra = " ".join(EXTRA_ARGS)
    # Activate the conda env, then hand off to run_ablation.sh. We source
    # conda.sh here (rather than in run_ablation.sh) so the same wrapper can
    # be run interactively without assuming a particular conda layout.
    return (
        f"source {CONDA_SH} && conda activate {CONDA_ENV} && "
        f"cd {PROJ_ROOT} && "
        f"bash {RUN_SCRIPT} {iter_size} {utd} {seed} {task_id} "
        f"--wandb_project {WANDB_PROJECT} {extra}".rstrip()
    )


def build_sbatch_line(job_name: str, payload: str) -> str:
    """Wrap ``payload`` (a shell string that may contain ``&``-separated
    parallel commands) in an sbatch --wrap invocation."""
    sbatch_opts = [
        f"-A {ACCOUNT}",
        f"-p {PARTITION}",
        f"--gres={GRES}",
        f"-N {NODES}",
        f"-n {NTASKS}",
        f"-c {CPUS_PER_TASK}",
        f"--qos=rail_gpu4_{PRIORITY}",
        f"-t {TIME_LIMIT}",
        f"--mem={MEM}",
        f"-J {job_name}",
        f"-o {PROJ_ROOT}/slurm_logs/{job_name}.%j.out",
        f"-e {PROJ_ROOT}/slurm_logs/{job_name}.%j.err",
        "--requeue",
        "--parsable",
    ]
    sbatch_cmd = "sbatch " + " ".join(sbatch_opts)
    # Escape single quotes in the payload so the outer --wrap '...' is safe.
    safe_payload = payload.replace("'", "'\\''")
    return f"{sbatch_cmd} --wrap '{safe_payload}'"


def main() -> None:
    combos = list(itertools.product(
        ITERATION_SIZES, MULTI_GRAD_STEPS, SEEDS, TASK_IDS
    ))
    print(
        f"Sweep: {len(ITERATION_SIZES)} iter_sizes x "
        f"{len(MULTI_GRAD_STEPS)} UTDs x {len(SEEDS)} seeds x "
        f"{len(TASK_IDS)} task_ids = {len(combos)} python runs"
    )

    # Group runs into slurm allocations of size NUM_PARALLEL.
    num_slurm_jobs = math.ceil(len(combos) / NUM_PARALLEL)
    print(f"Will submit {num_slurm_jobs} slurm jobs "
          f"({NUM_PARALLEL} python run(s) per slurm job)")

    log_dir = PROJ_ROOT / "slurm_logs"
    log_dir.mkdir(exist_ok=True)

    lines = [
        "#!/bin/bash",
        "# Auto-generated by launch_ablation.py. Do not edit by hand; re-run",
        "# launch_ablation.py to regenerate.",
        "set -euo pipefail",
        "",
    ]

    idx = 0
    for job_idx in range(0, len(combos), NUM_PARALLEL):
        batch = combos[job_idx : job_idx + NUM_PARALLEL]
        cmds = [build_single_command(*c) for c in batch]
        # Run parallel python jobs in background within the slurm alloc, then
        # wait. For NUM_PARALLEL==1 this is just a single foreground command.
        if len(cmds) == 1:
            payload = cmds[0]
        else:
            payload = " & ".join(f"({c})" for c in cmds) + " & wait"

        iter_s, utd_s, seed_s, task_s = batch[0]
        job_name = f"dsrl_t{task_s}_it{iter_s}_utd{utd_s}_s{seed_s}"
        if len(batch) > 1:
            job_name += f"_plus{len(batch)-1}"
        lines.append(f"# Job {idx}: {batch}")
        lines.append(
            f"jobid{idx}=$({build_sbatch_line(job_name, payload)}) && "
            f"echo \"submitted $jobid{idx}  ({job_name})\""
        )
        idx += 1

    OUTPUT_SBATCH.write_text("\n".join(lines) + "\n")
    os.chmod(OUTPUT_SBATCH, 0o755)
    print(f"Wrote {OUTPUT_SBATCH} ({idx} sbatch calls).")
    print(f"Priority = {PRIORITY}")
    print(f"Review with:  less {OUTPUT_SBATCH}")
    print(f"Submit with:  bash {OUTPUT_SBATCH}")


if __name__ == "__main__":
    main()
