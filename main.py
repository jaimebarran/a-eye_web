"""HPC segmentation job management.

Submits the nnU-Net inference job to the HPC cluster via Slurm, waits for
completion, and copies the resulting segmentation files back to the user's
local download folder.
"""

import os
import re
import subprocess
from datetime import datetime

from config import JOBFILE_HPC, JOBFILE_TEMPLATE, LOGS_FOLDER, OUTPUT_HPC, SSH_USER
from user_paths import UserPaths
from utils import (
    cancel_slurm_job,
    clean_email,
    copy_file_to_hpc,
    copy_files_from_hpc,
    copy_folder,
    modify_jobfile,
    move_file,
    print_and_log,
    user_reference_rmtree,
)

# ----------------------------------------------------------------------------------------------
# MAIN FUNCTION


def getSegmentation(
    user_email: str,
    paths: UserPaths,
) -> str:
    """Main function : performs inference on input folder and saves output in output folder

    Args:
        user_email (str): user's email address
        paths (UserPaths): the generated paths based on the user's address

    Returns:
        str: a message to announce segmentation done
    """
    # safe_email and timestamp are needed to find the HPC output folder
    # when copying the resulting files back to the local machine
    safe_email = clean_email(user_email)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobfile = paths.jobfile
    jobfile_hpc = f"{os.path.dirname(JOBFILE_HPC)}/nnunet_inference_{safe_email}.sh"

    modify_jobfile(JOBFILE_TEMPLATE, user_email, timestamp, jobfile, paths.hpc_input)
    copy_file_to_hpc(jobfile, os.path.dirname(jobfile_hpc))

    # Inference phase
    inference_command = f'ssh {SSH_USER} "sbatch --wait --partition=Dance --account=mattech --qos=normal {jobfile_hpc}"'

    process = subprocess.Popen(
        inference_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print_and_log("[A-eye] Submitted batch job to HPC...", "info", LOGS_FOLDER)

    # Getting the job ID at submission
    job_id: str = ""
    id_value: str = process.stdout.readline()
    found: re.Match | None = re.search(r"Submitted batch job (\d+)", id_value)

    if found:
        job_id = found.group(1)
        print_and_log(f"[A-eye] Slurm job ID = {job_id}", "info", LOGS_FOLDER)
        if job_id:
            paths.active_job_file.parent.mkdir(parents=True, exist_ok=True)
            paths.active_job_file.write_text(job_id)
    try:
        _, sub_stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, inference_command, stderr=sub_stderr
            )
    except Exception:
        if job_id:
            cancel_slurm_job(job_id)
        raise

    copy_files_from_hpc(f"{OUTPUT_HPC}/{safe_email}_{timestamp}", paths.download)
    # Logs handling
    output_logs = paths.download / "logs"
    if output_logs.exists():
        user_reference_rmtree(output_logs)
    output_logs.mkdir(parents=True, exist_ok=True)
    move_file(str(paths.download / "*.err"), output_logs)
    move_file(str(paths.download / "*.out"), output_logs)

    copy_folder(LOGS_FOLDER, output_logs)

    return "\nInference finished!!"
