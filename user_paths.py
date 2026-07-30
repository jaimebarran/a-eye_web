"""Dataclass for the paths at user-scope used in the segmentation tool.

This module contains the UserPaths dataclass. It gathers every local and HPC
path a user's segmentation tool use need.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserPaths:
    """User-scoped paths for all stages of the segmentation pipeline

    Attributes:
        upload:          Local path where user files are initially uploaded
        aux_base:        Local root directory for nnUNet computing files
        aux_input:       Local directory where nnUNet inference input data is staged
        output_zip:      Local path to the zipped archive of the inference results
        download:        Local directory containing the results ready for download
        jobfile:         Local path to the generated Bash script for job submission
        active_job_file: Local file storing the running Slurm job ID
        visualisation:   Local path to store visualisations generated, to be display on the plateform
        hpc_base_input:  Root directory on the HPC cluster for this user's input data
        hpc_input:       Subdirectory of hpc_base_input where input files are placed for inference
    """

    upload: Path
    aux_base: Path
    aux_input: Path
    output_zip: Path
    download: Path
    jobfile: Path
    active_job_file: Path
    visualisation: Path
    hpc_base_input: str
    hpc_input: str

    def create_directories(self) -> None:
        """Create the needed local directories if missing"""
        self.upload.mkdir(parents=True, exist_ok=True)
        self.aux_input.mkdir(parents=True, exist_ok=True)
        self.download.mkdir(parents=True, exist_ok=True)
        self.jobfile.parent.mkdir(parents=True, exist_ok=True)
        self.visualisation.mkdir(parents=True, exist_ok=True)
