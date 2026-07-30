"""Helper functions for the web platform.

This module contains everything the routes needs.
"""

import fnmatch
import glob
import gzip
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime
from functools import wraps
from os import path
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

import dicom2nifti
import nibabel
import numpy as np
import py7zr
import pycountry
import requests
from flask import current_app, jsonify, redirect, session, url_for
from flask_mail import Message

from app import mail
from config import (
    ALLOWED_EXTENSIONS,
    BASE_INPUT_HPC,
    DATA_FOLDER,
    LOGS_FOLDER,
    SSH_USER,
    STATS_FILE,
)
from user_paths import UserPaths

NOT_TO_OUTPUT = [
    "logs",
    "plans.pkl",
    "postprocessing.json",
    "prediction_time.txt",
    "*_cropped.nii.gz",
    "*_raw.nii.gz",
]


# Adapted from (open-webui-graphiti-memory, Skyzi000, accessed 01.06.2026)
# Line 472
# URL: https://github.com/Skyzi000/open-webui-graphiti-memory/blob/main/functions/action/add_graphiti_memory_action.py
def clean_email(email: str) -> str:
    """Clean the email by replacing @ and ."""
    return email.replace("@", "_at_").replace(".", "_")


def _header_to_dict(
    header: nibabel.nifti1.Nifti1Header,
) -> dict[str, list[float | int] | float | int | str | None]:
    """Serialize the extracted C structure (Nibabel)
       into a dict for the JSON message for frontend display

    Args:
        header (nibabel.nifti1.Nifti1Header): NIfTI header object returned by nibabel.load().header

    Returns:
        dict[str, list[float | int] | float | int | str | None]:
            JSON-serializable dict with header field name (key) and converted Python value
    """
    result: dict[str, list[float | int] | float | int | str | None] = {}

    for key, value in header.items():
        # datastructure conversion
        if isinstance(value, np.ndarray):
            value = value.tolist()
        # conversion to python type
        elif isinstance(value, (np.integer, np.floating)):
            value = value.item()
        # cleaning binary text
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", errors="replace").rstrip("\x00")
        # excluding invalid values (Nan, Inf)
        elif isinstance(value, float) and not np.isfinite(value):
            value = None
        # saving cleaned value
        result[key] = value
    return result


def extract_nifti_metadata(file_path: str) -> dict[str, dict]:
    """Extract metadata from a NIfTI file or archive

    Args:
        file_path (str): path of the file to extract from

    Returns:
        dict[str, dict]: case label, case header
                - on success : one entry per NIfTI file or DICOM series
                - on failure : empty dictionary
    """
    path = Path(file_path)
    extension = "".join(path.suffixes).lower()

    # handling single NIfTI file - no extraction nor conversion
    if extension in (".nii", ".nii.gz"):
        try:
            # Clean .gz extension natively using pathlib
            if path.suffix == ".gz":
                label = path.with_suffix("").name
            else:
                label = path.name
            return {label: _header_to_dict(nibabel.load(path).header)}
        except Exception:
            print_and_log(
                f"[A-eye] Error in single NIfTI extraction for {path.name}...",
                "error",
                LOGS_FOLDER,
            )
            return {}

    # Handle archives - need extraction and conversion (if DICOM series)
    if extension in (".zip", ".7z"):
        results: dict[str, dict] = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # extraction
            if extension == ".zip":
                with zipfile.ZipFile(path, "r") as zip_file:
                    zip_file.extractall(tmp_path)
            else:
                with py7zr.SevenZipFile(path, mode="r") as zip_file:
                    zip_file.extractall(path=tmp_path)

            converted_dir = tmp_path / "_converted_dcm"
            converted_dir.mkdir(exist_ok=True)

            # DICOM folders
            dicom_folders = {dcm.parent for dcm in tmp_path.rglob("*.dcm")}
            for folder in dicom_folders:
                rel_path = folder.relative_to(tmp_path)
                if rel_path == Path("."):
                    label = "<DICOM series>"
                    out_name = "_root"
                else:
                    label = f"{rel_path}/<DICOM series>"
                    out_name = str(rel_path).replace("/", "_")
                out_file = converted_dir / f"{out_name}.nii"
                try:
                    dicom2nifti.dicom_series_to_nifti(
                        str(folder), str(out_file), reorient_nifti=True
                    )
                    results[label] = _header_to_dict(nibabel.load(out_file).header)
                except Exception:
                    print_and_log(
                        f"[A-eye] Error in DICOM processing in {folder.name}...",
                        "error",
                        LOGS_FOLDER,
                    )

            # for NIfTI files
            for nifti_file in (*tmp_path.rglob("*.nii"), *tmp_path.rglob("*.nii.gz")):
                if converted_dir in nifti_file.parents:
                    continue
                try:
                    if nifti_file.suffix == ".gz":
                        clean_file = nifti_file.with_suffix("")
                    else:
                        clean_file = nifti_file

                    label = str(clean_file.relative_to(tmp_path))
                    results[label] = _header_to_dict(nibabel.load(nifti_file).header)

                except Exception:
                    print_and_log(
                        f"[A-eye] Error in NIfTI in folder extraction for {nifti_file.name}...",
                        "error",
                        LOGS_FOLDER,
                    )

        return results

    return {}


def cancel_slurm_job(id_job: str) -> None:
    """Cancel a slurm job

    Args:
        id_job (str): id of the slurm job (string as is from a linux env)
    """
    print_and_log(
        f"[A-eye] Cancelling the slurm job {id_job}...", "warning", LOGS_FOLDER
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            SSH_USER,
            f"scancel {id_job}",
        ],
        check=True,
        timeout=20,
    )


def clear_folder(folder: Path | str) -> None:
    """Deletes all files and subdirectories inside folder"""
    delete_files_in_folder(folder)
    delete_subfolders(folder)


def get_user_paths(user_email: str) -> UserPaths:
    """Generate all needed local and HPC paths for a user, based on his email

    Args:
        user_email (str): The email address of the user to process

    Returns:
        UserPaths: Dataclass holding the different paths relative to the user
    """
    cleaned_email: str = clean_email(user_email)
    base_path: Path = Path("./nnUNet/nnUNet_inference") / cleaned_email

    return UserPaths(
        upload=Path("./static/upload") / cleaned_email,
        aux_base=base_path,
        aux_input=base_path / "input",
        output_zip=base_path / "output.zip",
        download=base_path / "output",
        jobfile=Path("./jobfiles") / f"nnunet_inference_{cleaned_email}.sh",
        active_job_file=Path("./static/active_jobs") / f"{cleaned_email}.txt",
        visualisation=base_path / "visualisation",
        hpc_base_input=f"{BASE_INPUT_HPC}/{cleaned_email}",
        hpc_input=f"{BASE_INPUT_HPC}/{cleaned_email}/input",
    )


def allowed_file(filename):
    """Whether the file type is allowed or not (allowed type : .nii.gz / .zip /
    .7z / .nii)

    Args:
        filename (str): The name of the file

    Returns:
        bool: True if allowed, else False
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def zip_folder(folder_path, output_path, to_exclude=NOT_TO_OUTPUT):
    """Zip a folder, excluding files/directories not to output

    Args:
        folder_path (str): path of the folder to zip
        output_path (str): path of the resulting .zip archive
        to_exclude (list[str], optional): glob patterns of files/directories to leave
            out of the archive.
    """
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, directories, files in os.walk(folder_path):
            # remove directories/files to exclude from output
            for directory in list(directories):
                if any(fnmatch.fnmatch(directory, p) for p in to_exclude):
                    directories.remove(directory)

            for file in files:
                if any(fnmatch.fnmatch(file, p) for p in to_exclude):
                    continue

                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), folder_path),
                )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("routes.login"))
        return f(*args, **kwargs)

    return decorated


def requires_auth_api(f):
    """Auth guard for JSON/XHR endpoints: returns 401 JSON instead of redirecting."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return jsonify({"message": "Unauthorized", "status": "error"}), 401
        return f(*args, **kwargs)

    return decorated


def get_country_from_ip(ip):
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    data = response.json()
    country = data.get("country")
    print_and_log(f"[A-eye] IP: {ip}, Country: {country}", "info", LOGS_FOLDER)
    return country


def convert_country_to_iso3(country_code):
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.alpha_3
    except AttributeError:
        return None


def copy_segmentation_data(user_email, output):
    zurich_time = datetime.now(ZoneInfo("Europe/Zurich"))
    timestamp = zurich_time.strftime("%Y%m%d_%H%M")
    user_reference_email = clean_email(user_email)
    dest_dir = f"{DATA_FOLDER}/{user_reference_email}_{timestamp}"

    os.makedirs(dest_dir, exist_ok=True)

    output_dest = os.path.join(dest_dir, "output")

    copy_folder(output, output_dest)

    print_and_log(
        f"[A-eye] Copied segmentation data to {dest_dir}", "info", LOGS_FOLDER
    )
    sync_logs_to_output(output)


def unzip_file(file_type, source, destination):
    try:
        os.makedirs(destination, exist_ok=True)

        if file_type == "zip":
            print_and_log(f"[A-eye] Unzipping ZIP file: {source}", "info", LOGS_FOLDER)
            with zipfile.ZipFile(source, "r") as zip_ref:
                abs_dest = os.path.abspath(destination)

                for member in zip_ref.namelist():
                    member_path = os.path.abspath(os.path.join(destination, member))
                    if not os.path.commonpath([abs_dest, member_path]) == abs_dest:
                        raise Exception(f"Unuser_reference path in zip file: {member}")

                zip_ref.extractall(destination)

        elif file_type == "7z":
            print_and_log(f"[A-eye] Unzipping 7z file: {source}", "info", LOGS_FOLDER)
            with py7zr.SevenZipFile(source, mode="r") as archive:
                archive.extractall(path=destination)

        else:
            raise ValueError(f"Unsupported archive type: {file_type}")

    except Exception as e:
        print_and_log(
            f"[A-eye] Error unzipping file {source}: {e}", "error", LOGS_FOLDER
        )
        raise


def copy_folder(source, destination):
    os.makedirs(destination, exist_ok=True)
    for item in os.listdir(source):
        if fnmatch.fnmatch(item, "*_raw.nii.gz"):
            continue
        s = os.path.join(source, item)
        d = os.path.join(destination, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def copy_clean_files(src_folder, dst_folder):
    """
    Copy only *_0000.nii.gz files from src_folder (and subfolders)
    into dst_folder (flat, no subfolder structure).
    """
    os.makedirs(dst_folder, exist_ok=True)

    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.endswith("_0000.nii.gz"):
                src_path = os.path.join(root, f)
                dest_path = os.path.join(dst_folder, f)

                # Avoid overwriting if two files have the same name
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(f)
                    dest_path = os.path.join(dst_folder, f"{base}_dup{ext}")

                shutil.copy2(src_path, dest_path)


def copy_file(source, destination):
    os.makedirs(destination, exist_ok=True)
    shutil.copy(source, destination)


def move_file(pattern, destination):
    os.makedirs(destination, exist_ok=True)
    for file_path in glob.glob(pattern):
        shutil.move(file_path, destination)


def copy_file_to_hpc(source, destination):
    """Copy a single local file to the HPC

    Args:
        source (str): source path of the file to copy
        destination (str): destination path on the HPC
    """
    print_and_log(
        f"[A-eye] Copying files from local {source} to HPC {destination}...",
        "info",
        LOGS_FOLDER,
    )
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=8", source, f"{SSH_USER}:{destination}"],
        check=True,
        timeout=20,
    )


def copy_folder_to_hpc(source: str, destination: str) -> None:
    """Copy a local folder to the HPC

    Args:
        source (str): source path of the folder to copy
        destination (str): destination path on the HPC
    """
    print_and_log(
        f"[A-eye] Copying files from local {source} to HPC {destination}...",
        "info",
        LOGS_FOLDER,
    )
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=8", "-r", source, f"{SSH_USER}:{destination}"],
        check=True,
        timeout=30,
    )


def copy_files_from_hpc(source, destination):
    """Copy the content of a HPC folder to a local folder

    Args:
        source (str): source folder path on the HPC
        destination (str): local destination folder path, created if missing
    """
    print_and_log(
        f"[A-eye] Copying files from HPC {source} to local {destination}...",
        "info",
        LOGS_FOLDER,
    )
    os.makedirs(destination, exist_ok=True)
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=8", "-r", f"{SSH_USER}:{source}/.", destination],
        check=True,
        timeout=30,
    )


def delete_files_in_folder(folder):
    if not os.path.exists(folder):
        return

    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                user_reference_remove_file(file_path)
            except Exception as e:
                print_and_log(
                    f"[A-eye] Failed to delete {file_path}. Reason: {e}",
                    "error",
                    LOGS_FOLDER,
                )


def clean_folder_hpc(folder: str) -> None:
    """Clear all files and subfolders of a given folder on the HPC

    Args:
        folder (str): path of the folder to clean
    """
    print_and_log(f"[A-eye] Cleaning folder {folder} on HPC...", "info", LOGS_FOLDER)
    subprocess.run(
        [
            "ssh",  # using the ssh client
            "-o",  # ssh config option n.1
            "BatchMode=yes",  # non-interactive mode : will fail if a password if asked
            "-o",  # ssh config option n.2
            "ConnectTimeout=8",  # Maximum time (in seconds) to establish the ssh connection
            SSH_USER,  # ID and address of the distant server
            f"mkdir -p {folder} && rm -rf {folder}/*",
        ],  # Commands executed on the server : create folders and clear it's content
        check=True,  # Raise a CalledProcessError if exit code isn't 0
        timeout=20,  # Raise a TimeOutException if execution time is more than the set value (in seconds)
    )
    print_and_log(f"[A-eye] Folder {folder} cleaned successfully.", "info", LOGS_FOLDER)


def delete_folder(folder):
    user_reference_rmtree(folder)


def delete_subfolders(folder):
    if not os.path.exists(folder):
        return

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            try:
                user_reference_rmtree(item_path)
            except Exception as e:
                print_and_log(
                    f"[A-eye] Failed to delete {item_path}. Reason: {e}",
                    "error",
                    LOGS_FOLDER,
                )


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(
            path,
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
        )
        func(path)
    except Exception:
        raise exc_info[1]


def user_reference_remove_file(path):
    if not os.path.exists(path):
        return

    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    except Exception:
        pass

    os.remove(path)


def user_reference_rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path, onerror=_handle_remove_readonly)


def check_dicom_folders_names(folder):
    # Get a list of all DICOM folders in the input folder
    dicom_folders = find_dicom_folders(folder)
    # Check dicom folders names
    if not dicom_folders:
        print_and_log("[A-eye] No DICOM folders found.", "info", LOGS_FOLDER)
        return
    else:
        print_and_log("[A-eye] Checking DICOM folders names...", "info", LOGS_FOLDER)
        for dicom_folder in dicom_folders:
            dicom_folder_name = os.path.basename(dicom_folder)
            parent_folder_path = os.path.dirname(dicom_folder)
            parent_folder_name = os.path.basename(parent_folder_path)
            # Check if dicom_folder name already starts with parent_folder_name
            if not dicom_folder_name.startswith(parent_folder_name):
                new_folder_name = parent_folder_name + "_" + dicom_folder_name
                new_folder_path = os.path.join(parent_folder_path, new_folder_name)
                os.rename(dicom_folder, new_folder_path)
            # Convert to nifti
            convert_to_nifti(folder)


def check_nifti_files(folder):
    for file_path in glob.glob(os.path.join(folder, "**", "*.nii"), recursive=True):
        gz_path = file_path + ".gz"
        with open(file_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)


def check_filenames(folder):
    for file_path in glob.glob(
        os.path.join(folder, "**", "*"), recursive=True
    ):  # catch any case; we'll filter below
        # only operate on files that end with .nii.gz (case-insensitive)
        if not file_path.lower().endswith(".nii.gz"):
            continue

        original_basename = os.path.basename(file_path)  # e.g. "x_...1.000.nii.gz"
        base_no_ext = original_basename[: -len(".nii.gz")]  # e.g. "x_...1.000"
        file_extension = ".nii.gz"

        print_and_log(f"[A-eye] file name: {original_basename}", "info", LOGS_FOLDER)
        print_and_log(f"[A-eye] file base (no ext): {base_no_ext}", "info", LOGS_FOLDER)
        print_and_log(f"[A-eye] file extension: {file_extension}", "info", LOGS_FOLDER)
        print_and_log(f"[A-eye] absolute file path: {file_path}", "info", LOGS_FOLDER)

        if not base_no_ext.endswith("_0000"):
            # pass base without extension so correct_filename can build: base_no_ext + '_0000' + '.nii.gz'
            correct_filename(file_path, base_no_ext, file_extension)


def correct_filename(file_path, file_name, file_extension):
    print_and_log("[A-eye] Changing filename to nnUNet format...", "info", LOGS_FOLDER)
    new_file_name = f"{file_name}_0000{file_extension}"  # extension for nnUNet
    print_and_log(f"[A-eye] New filename = {new_file_name}", "info", LOGS_FOLDER)
    os.rename(file_path, os.path.join(os.path.dirname(file_path), new_file_name))


def convert_to_nifti(folder):
    # Get a list of all DICOM folders in the input folder
    dicom_folders = find_dicom_folders(folder)
    if len(dicom_folders) > 0:
        print_and_log(
            "[A-eye] Converting DICOM to NIfTI format...", "info", LOGS_FOLDER
        )
        # Convert each DICOM folder to NIfTI format
        for dicom_folder in dicom_folders:
            filename = str(os.path.basename(dicom_folder) + ".nii.gz")
            dicom2nifti.dicom_series_to_nifti(
                dicom_folder, f"{folder}/{filename}", reorient_nifti=True
            )
            # cmd = ["dcm2niix", "-f", filename, "-z", "y", "-o", output_nifti_folder, input_dicom_folder]
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
            # _ = process.communicate()[0]  # the [0] is to return just the output,
            #  because otherwise it would be outs, errs = proc.communicate()
            # delete_folder(folder)  # Delete the original DICOM folder after conversion


def find_dicom_folders(root_path):
    dicom_folders = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if fnmatch.fnmatch(filename, "*.dcm"):
                dicom_folders.append(dirpath)
                break

    return dicom_folders


def start_docker():
    # Check if Docker is running and if not, initialize it
    try:
        # docker version
        run_command_and_print_output("docker version")
        print_and_log("\n[A-eye] Docker is already running...", "info", LOGS_FOLDER)
    except:
        # If Docker is not running...
        print_and_log("[A-eye] Docker was not initialized!!", "warning", LOGS_FOLDER)
        # ... start it!
        print_and_log("[A-eye] Initializing docker...", "info", LOGS_FOLDER)
        # docker start
        run_command_and_print_output("systemctl start docker")
        # sleep 1s
        run_command_and_print_output("sleep 1")
        print_and_log("[A-eye] Docker has been started", "info", LOGS_FOLDER)


def clear_logs(logs_folder=None):
    open(f"{logs_folder}/app.log", "w").close()
    open(f"{logs_folder}/console.log", "w").close()


def prepare_log_target(target):
    os.makedirs(target, exist_ok=True)

    try:
        os.chmod(target, 0o775)
    except PermissionError:
        pass

    for filename in ("app.log", "console.log"):
        path = os.path.join(target, filename)
        if os.path.exists(path):
            try:
                os.chmod(path, 0o664)
            except PermissionError:
                pass


def get_log_targets(logs_folder=None):
    prepare_log_target(logs_folder)
    return [logs_folder]


def append_log_line(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{line}\n")


def sync_logs_to_output(output_folder: str | Path) -> None:
    """Copy the current app and console logs a subfolder of the output folder

    Args:
        output_folder (str | Path): path were the logs subfolder will be made
    """
    output_logs = os.path.join(output_folder, "logs")
    prepare_log_target(output_logs)

    for filename in ("app.log", "console.log"):
        source = os.path.join(LOGS_FOLDER, filename)
        destination = os.path.join(output_logs, filename)
        if os.path.exists(source):
            try:
                if os.path.exists(destination):
                    os.remove(destination)
                shutil.copy2(source, destination)
            except PermissionError:
                pass


def print_console(text=None, logs_folder=None):
    targets = get_log_targets(logs_folder)

    for index, target in enumerate(targets):
        logs_file = f"{target}/console.log"
        try:
            append_log_line(logs_file, text)
        except PermissionError:
            if index == 0:
                raise


def print_and_log(text=None, level="info", logs_folder=None):
    timestamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d %H:%M:%S")
    app_line = f"{timestamp} {level.upper()} {text}"
    targets = get_log_targets(logs_folder)

    for index, target in enumerate(targets):
        console_file = f"{target}/console.log"
        app_file = f"{target}/app.log"

        try:
            append_log_line(console_file, text)
            append_log_line(app_file, app_line)
        except PermissionError:
            if index == 0:
                raise


def run_command_and_print_output(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        shell=True,
    )
    console_output, console_errors = process.communicate()

    if console_output:
        console_output = console_output.decode("utf-8")
        for line in console_output.splitlines():
            print_console(line, LOGS_FOLDER)

    if console_errors:
        console_errors = console_errors.decode("utf-8")
        for line in console_errors.splitlines():
            print_console(line, LOGS_FOLDER)


def clean_folders(user_email: str) -> None:
    """Clear all folders used in the segmentation process from old remaining data"""
    paths = get_user_paths(user_email)
    paths.create_directories()

    clear_folder(paths.upload)
    clear_folder(paths.aux_input)
    delete_files_in_folder(paths.aux_base)

    user_reference_rmtree(paths.visualisation)
    paths.visualisation.mkdir(parents=True, exist_ok=True)
    user_reference_rmtree(paths.download)
    paths.download.mkdir(parents=True, exist_ok=True)
    paths.jobfile.unlink(missing_ok=True)

    clean_folder_hpc(paths.hpc_input)


def get_management_api_token():
    url = f"https://{current_app.config['AUTH0_DOMAIN']}/oauth/token"
    payload = {
        "client_id": current_app.config["AUTH0_CLIENT_ID"],
        "client_secret": current_app.config["AUTH0_CLIENT_SECRET"],
        "audience": f"https://{current_app.config['AUTH0_DOMAIN']}/api/v2/",
        "grant_type": "client_credentials",
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["access_token"]


def get_user_data():
    auth0_domain = current_app.config["AUTH0_DOMAIN"]
    access_token = get_management_api_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    # Include fields you want explicitly
    params = {
        "fields": "email,last_ip",
        "include_fields": "true",
        "per_page": 100,
        "page": 0,
    }

    all_users = []
    while True:
        response = requests.get(
            f"https://{auth0_domain}/api/v2/users", headers=headers, params=params
        )
        response.raise_for_status()
        page_users = response.json()
        if not page_users:
            break
        all_users.extend(page_users)
        params["page"] += 1

    return all_users


# def send_email_async(to, body):
#     threading.Thread(target=send_email, args=(to, body)).start()


def send_email(to, body):
    with current_app.app_context():
        msg = Message(subject="Segmentation Task Completed", recipients=[to], body=body)
        mail.send(msg)


def load_stats():
    if not os.path.exists(STATS_FILE):
        return {"cases_processed": 0}
    with open(STATS_FILE, "r") as f:
        return json.load(f)


def increment_cases_processed():
    stats = load_stats()
    stats["cases_processed"] += 1
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


def get_cases_processed():
    return load_stats()["cases_processed"]


def modify_jobfile(
    template_file: str | Path,
    user_email: str,
    timestamp: str,
    output_file: str | Path,
    hpc_input: str,
) -> None:
    """Modify the jobfile template with user-specific information"""

    # Make the email user_reference for paths
    user_reference_email = clean_email(user_email)

    # Base results dir
    base_results_dir = "/home/jaime.barrancohernandez/results/nnunet"

    # Create unique folder for this user + timestamp
    user_dir = os.path.join(base_results_dir, f"{user_reference_email}_{timestamp}")
    os.system(f"ssh {SSH_USER} 'mkdir -p {user_dir}'")

    # Read the template
    with open(template_file, "r") as f:
        template = Template(f.read())
        job_script = template.safe_substitute(MAIL_USER=user_email)

    # Update SBATCH output/error paths
    out_file = (
        f"{user_dir}/{user_reference_email}_{timestamp}_nnUNet_predict.%N.%j.%a.out"
    )
    err_file = (
        f"{user_dir}/{user_reference_email}_{timestamp}_nnUNet_predict.%N.%j.%a.err"
    )
    job_script = re.sub(r"(#SBATCH --output=).+\.out", rf"\1{out_file}", job_script)
    job_script = re.sub(r"(#SBATCH --error=).+\.err", rf"\1{err_file}", job_script)

    # Update the specific --bind line
    job_script = re.sub(
        r"--bind\s+/home/jaime\.barrancohernandez/results/nnunet:/output",
        f"--bind {user_dir}:/output",
        job_script,
    )
    # isolate HPC input bind
    job_script = re.sub(
        r"--bind\s+\S+:/input", f"--bind {hpc_input}:/input", job_script
    )

    # Write modified jobfile
    with open(output_file, "w") as f:
        f.write(job_script)


def upload_files(upload_folder: str, aux_input: str, hpc_base_input: str) -> None:
    """Upload files into HPC folder after some operations
        1. Unzip if needed
        2. If .dcm, then convert to .nii
        3. Compress .nii to nii.gz
        4. Rename to expected format (*_0000.nii.gz) for HPC
        5. Copy the results into the HPC folder

    Args:
        upload_folder (String):     Local path where user files are initially uploaded
        aux_input (String):         Local directory where nnUNet inference input data is staged
        hpc_base_input (String):    Root directory on the HPC cluster for this user's input data
    """
    # paths
    aux_in = aux_input
    base_input_hpc = hpc_base_input  # input folder on HPC

    # 1. Check if input folder contains zip/7z files and unzip them
    if os.path.isdir(upload_folder):
        for file in os.listdir(upload_folder):
            fpath = os.path.join(upload_folder, file)
            if os.path.isfile(fpath):
                ext = os.path.splitext(file)[1].lower()
                if ext in [".zip", ".7z"]:
                    unzip_file(
                        ext[1:], fpath, upload_folder
                    )  # unzip into the same input folder

    # 2. Check dicom folders
    check_dicom_folders_names(upload_folder)

    # 3. Check nifti files
    check_nifti_files(upload_folder)

    # 4. Check filenames
    check_filenames(upload_folder)

    # 5. Copy the final results to aux_in
    copy_clean_files(upload_folder, aux_in)

    # 6. Copy aux_in to base_input_hpc
    copy_folder_to_hpc(aux_in, base_input_hpc)
