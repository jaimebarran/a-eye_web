"""Flask routes for the A-eye web platform.
"""

import csv
import math
import os
import re
import secrets
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus, urlencode

import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import plotly.express as px
from authlib.integrations.base_client.errors import MismatchingStateError, OAuthError
from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from app import oauth
from config import LOGS_FOLDER
from main import getSegmentation
from package.biomarkers.biomarkers import (
    compute_axial_length_data,
    compute_volumetry,
    extract_axial_length_measurements,
)
from package.biomarkers.visualisations import plot_axial_length
from package.quadrant_segmentation.quadrant import (
    crop_quadrant,
    merge_quadrants,
    uncrop_quadrant,
)
from package.statistical_analysis.analysis import (
    load_reference,
    references_iqr_bounds,
    references_means,
    references_size,
    references_standard_deviation,
)
from package.statistical_analysis.visualisations import (
    plot_axial_length_violin,
    plot_volumetry_violin,
)
from user_paths import UserPaths
from utils import (
    Message,
    allowed_file,
    cancel_slurm_job,
    clean_folder_hpc,
    clean_folders,
    convert_country_to_iso3,
    copy_folder_to_hpc,
    copy_segmentation_data,
    extract_nifti_metadata,
    get_cases_processed,
    get_country_from_ip,
    get_user_data,
    get_user_paths,
    increment_cases_processed,
    mail,
    print_and_log,
    requires_auth,
    requires_auth_api,
    sync_logs_to_output,
    upload_files,
    zip_folder,
)


def _process_eye(
    paths: UserPaths,
    case_name: str,
    side: str,
    left_side: bool,
    raw_path: Path,
) -> tuple[dict, dict | None]:
    """Compute and make the visualisation for biomarkers for one eye.

    Args:
        paths (UserPaths): user paths
        case_name (str): name of the case
        side (str): "left" or "right"
        left_side (bool): True if processing the left eye, False for the right
        raw_path (Path): path to the raw NIfTI file

    Returns:
        tuple[dict, dict | None]: biomarkers values, biomarkers values formatted for csv
    """
    segmentation_path = paths.download / f"{case_name}_{side}_cropped.nii.gz"

    segmentation_image = nib.load(str(segmentation_path))
    raw_image = crop_quadrant(raw_path, left_side=left_side)

    volumes = compute_volumetry(segmentation_image)
    axial_data = compute_axial_length_data(segmentation_image, raw_image)
    axial_measures = extract_axial_length_measurements(axial_data)
    # get the visualisation of AL
    fig = plot_axial_length(axial_data, case_name=f"{case_name} ({side} eye)")
    if fig is not None:
        img_filename = f"{case_name}_{side}_axial_length.png"
        fig.savefig(paths.download / img_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        img_url = f"/result-image/{img_filename}"
    else:
        img_url = None

    eye_data = {**volumes, **axial_measures, "axial_length_image": img_url}
    # convert the nan to None to avoid json error
    for key in eye_data:
        if key != "axial_length_image" and math.isnan(eye_data[key]):
            eye_data[key] = None

    # add the references and outliers
    ref_means = references_means(side)
    ref_standard_deviation = references_standard_deviation(side)
    bounds = {
        "F": references_iqr_bounds(side, "F"),
        "M": references_iqr_bounds(side, "M"),
    }
    eye_data["reference_mean"] = {}
    eye_data["reference_std"] = {}
    eye_data["reference_count"] = references_size(side)
    eye_data["outliers"] = {"F": {}, "M": {}}

    for key, value in ref_means.items():
        if key in eye_data:
            eye_data["reference_mean"][key] = value
            eye_data["reference_std"][key] = ref_standard_deviation[key]
            case_value = eye_data[key]
            if case_value is not None:
                for sex in ("F", "M"):
                    lower_bound, upper_bound = bounds[sex][key]
                    eye_data["outliers"][sex][key] = (
                        case_value < lower_bound or case_value > upper_bound
                    )

    # add the violin plot with this case's own value marked on it
    vol_violin_filename = f"{case_name}_{side}_vol_violin_plot.png"
    vol_violin_path = paths.visualisation / vol_violin_filename
    vol_reference = load_reference(side, volumetry=True)
    violin_fig = plot_volumetry_violin(vol_reference, volumes)
    violin_fig.savefig(vol_violin_path, dpi=150, bbox_inches="tight")
    plt.close(violin_fig)
    eye_data["vol_violin_image"] = f"/display-image/{vol_violin_filename}"

    al_violin_filename = f"{case_name}_{side}_al_violin_plot.png"
    al_violin_path = paths.visualisation / al_violin_filename
    al_reference = load_reference(side, axial_length=True)
    al_violin_fig = plot_axial_length_violin(al_reference, axial_measures)
    al_violin_fig.savefig(al_violin_path, dpi=150, bbox_inches="tight")
    plt.close(al_violin_fig)
    eye_data["al_violin_image"] = f"/display-image/{al_violin_filename}"

    csv_row = {"case": case_name, "side": side, **volumes, **axial_measures}
    return eye_data, csv_row


def _cancel_job(paths: UserPaths) -> None:
    """Internal function canceling Slurm job

    Args:
        paths (UserPaths): initialized paths for the concerned user
    """
    if not paths.active_job_file.exists():
        return
    job_id = paths.active_job_file.read_text().strip()
    paths.active_job_file.unlink(missing_ok=True)

    if job_id:
        print_and_log(
            f"[A-eye] Cancelling Slurm job {job_id}...", "warning", LOGS_FOLDER
        )
        try:
            cancel_slurm_job(job_id)
        except Exception:
            current_app.logger.exception("Failed to cancel Slurm job %s", job_id)


bp = Blueprint("routes", __name__)

high_scale_nb_users = 100


@bp.route("/")
def welcome():
    try:
        cases_processed = get_cases_processed()
    except Exception:
        current_app.logger.exception("Could not read cases_processed from stats file")
        cases_processed = 0
    return render_template("welcomepage.html", cases_processed=cases_processed)


@bp.route("/callback", methods=["GET", "POST"])
def callback():
    try:
        token = oauth.auth0.authorize_access_token()
        state = session.pop("state", None)
        if request.args.get("state") != state:
            raise MismatchingStateError()
        nonce = session.pop("nonce", None)
        userinfo = oauth.auth0.parse_id_token(token, nonce=nonce)

        if not userinfo.get("email_verified"):
            return redirect(url_for("routes.verify_email"))

        session["user"] = userinfo  # stores user in session
        return redirect(url_for("routes.segmentation"))
    except OAuthError as error:
        flash("Authentication failed: " + error.description)
        return redirect(url_for("routes.welcome"))


@bp.route("/verify_email")
def verify_email():
    return render_template("verify_email.html")


@bp.route("/login")
def login():
    nonce = secrets.token_urlsafe()
    state = secrets.token_urlsafe()
    session["nonce"] = nonce
    session["state"] = state
    redirect_uri = current_app.config.get("AUTH0_CALLBACK_URL") or url_for(
        "routes.callback", _external=True
    )
    return oauth.auth0.authorize_redirect(
        redirect_uri=redirect_uri, nonce=nonce, state=state
    )


@bp.route("/logout")
def logout():
    session.clear()
    return_to = current_app.config.get("AUTH0_LOGOUT_URL") or url_for(
        "routes.welcome", _external=True
    )
    return redirect(
        "https://"
        + current_app.config["AUTH0_DOMAIN"]
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": return_to,
                "client_id": current_app.config["AUTH0_CLIENT_ID"],
            },
            quote_via=quote_plus,
        )
    )


@bp.route("/users")
def users():
    users_data = get_user_data()
    total_users = len(users_data)
    cases_processed = get_cases_processed()

    # Calculate the number of institutions
    domains = set()
    for user in users_data:
        email = user.get("email")
        if email:
            domain = email.split("@")[-1]
            domains.add(domain)
    total_institutions = len(domains)

    # Get country data from user IPs
    country_counts = {}
    for user in users_data:
        if isinstance(user, dict):
            ip = user.get("last_ip")
            if ip:
                country = get_country_from_ip(ip)
                if country:
                    country_iso3 = convert_country_to_iso3(country)
                    if country_iso3:
                        if country_iso3 in country_counts:
                            country_counts[country_iso3] += 1
                        else:
                            country_counts[country_iso3] = 1

    # Create a DataFrame for the country data
    df = pd.DataFrame(list(country_counts.items()), columns=["Country", "Count"])

    # Generate the choropleth map
    fig = px.choropleth(
        df,
        locations="Country",
        locationmode="ISO-3",
        color="Count",
        color_continuous_scale="Greens",
        range_color=(0, high_scale_nb_users),
    )

    fig.update_layout(
        autosize=True,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        dragmode=False,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="equirectangular",
            lataxis=dict(range=[-90, 90]),
            lonaxis=dict(range=[-180, 180]),
            center=dict(lat=0, lon=0),
        ),
        coloraxis_showscale=False,
    )

    map_html = fig.to_html(full_html=False, config={"responsive": True})

    return render_template(
        "users.html",
        total_users=total_users,
        cases_processed=cases_processed,
        total_institutions=total_institutions,
        map_html=map_html,
    )


@bp.route("/about")
def about():
    return render_template("about.html")


@bp.route("/faq")
def faq():
    return render_template("faq.html")


@bp.route("/segmentation")
def segmentation():
    if "user" in session:
        user_email: str = session.get("user", {}).get("email", "unknown_user")
        _cancel_job(get_user_paths(user_email))
        return render_template("segmentation.html")
    return redirect(url_for("routes.login"))


@bp.route("/upload", methods=["POST"])
@requires_auth_api
def upload_file() -> tuple[Response, int]:
    """File upload handling for the user logged-in

    1. clear the user's previous data
    2. saves provided files locally
    3. copy them to the HPC input directory

    Returns:
        tuple[Response, int]: JSON response with upload and HTTP status code
    """
    user_email: str = session.get("user", {}).get("email", "unknown_user")
    paths: UserPaths = get_user_paths(user_email)

    _cancel_job(paths)

    if "files[]" not in request.files:
        return jsonify({"message": "No file found", "status": "error"}), 400

    files = request.files.getlist("files[]")
    uploaded_files = []
    rejected_files = []

    try:
        # 1. clear the user's previous data
        clean_folders(user_email)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        current_app.logger.exception("Could not prepare HPC folders for upload")
        return jsonify({
            "message": "Could not reach the HPC over SSH to prepare the upload."
            "Please try again once chacha/disco SSH access is available.",
            "status": "error",
            "error": str(error),
        }), 503

    # 2. Save provided files locally and extract their metadata
    metadata: dict[str, dict] = {}
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # werkzeug sanitisation
            filepath = paths.upload / filename
            file.save(filepath)
            uploaded_files.append(filename)
            metadata[filename] = extract_nifti_metadata(str(filepath))
        else:
            rejected_files.append(file.filename)

    try:
        # 3. copy the provided files in the HPC input directory
        upload_files(paths.upload, paths.aux_input, paths.hpc_base_input)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        current_app.logger.exception("Could not copy uploaded files to HPC")
        return jsonify({
            "message": "Files were received locally,"
            " but could not be copied to the HPC over SSH."
            " Please try again once chacha/disco SSH access is available.",
            "status": "error",
            "error": str(error),
        }), 503

    if uploaded_files:
        message = f"Uploaded: {', '.join(uploaded_files)}"
        if rejected_files:
            message += f" | Rejected: {', '.join(rejected_files)}"
        return jsonify({
            "message": message,
            "status": "success",
            "metadata": metadata,
        }), 200
    else:
        return jsonify({"message": "No valid files uploaded", "status": "error"}), 400


@bp.route("/segment", methods=["POST"])
@requires_auth_api
def segment() -> tuple[Response, int]:
    """Run the segmentation pipeline for the logged-in user.

    1. submits the HPC job and waits for completion
    2. zips the output if existing
    3. starts a background thread to copy results to Filer01

    Returns:
        tuple[Response, int]: JSON response with status message and code.
        200 on success - 500 if the pipeline raised error
    """
    user_email: str = session.get("user", {}).get("email", "unknown_user")
    paths: UserPaths = get_user_paths(user_email)

    _cancel_job(paths)

    # keeping the original dimension of the file
    original_shapes: dict[str, tuple] = {}

    for original_file in paths.aux_input.glob("*_0000.nii.gz"):
        case_name = original_file.name.replace("_0000.nii.gz", "")
        original_shapes[case_name] = nib.load(original_file).shape
        # copy original file for overlay visualization of results
        shutil.copy2(original_file, paths.download / f"{case_name}_raw.nii.gz")

        left_cropped_img = crop_quadrant(original_file, left_side=True)
        right_cropped_img = crop_quadrant(original_file, left_side=False)
        # back to HPC as cropped version
        nib.save(left_cropped_img, paths.aux_input / f"{case_name}_left_0000.nii.gz")
        nib.save(right_cropped_img, paths.aux_input / f"{case_name}_right_0000.nii.gz")

        original_file.unlink()
    # to clean it from the original
    clean_folder_hpc(paths.hpc_input)
    # to add it the cropped versions
    copy_folder_to_hpc(str(paths.aux_input), paths.hpc_base_input)

    try:
        getSegmentation(user_email, paths)
    except Exception as error:
        paths.active_job_file.unlink(missing_ok=True)
        print_and_log(f"[A-eye] Segmentation failed: {error}", "error", LOGS_FOLDER)
        sync_logs_to_output(paths.download)
        clean_folders(user_email)
        return jsonify({"message": "Segmentation failed", "error": str(error)}), 500

    paths.active_job_file.unlink(missing_ok=True)

    for case_name, original_shape in original_shapes.items():
        left_segmented = nib.load(paths.download / f"{case_name}_left.nii.gz")
        right_segmented = nib.load(paths.download / f"{case_name}_right.nii.gz")

        shutil.copy2(
            paths.download / f"{case_name}_left.nii.gz",
            paths.download / f"{case_name}_left_cropped.nii.gz",
        )
        shutil.copy2(
            paths.download / f"{case_name}_right.nii.gz",
            paths.download / f"{case_name}_right_cropped.nii.gz",
        )

        left_uncropped = uncrop_quadrant(left_segmented, original_shape, left_side=True)
        right_uncropped = uncrop_quadrant(
            right_segmented, original_shape, left_side=False
        )

        nib.save(left_uncropped, paths.download / f"{case_name}_left.nii.gz")
        nib.save(right_uncropped, paths.download / f"{case_name}_right.nii.gz")

        merged = merge_quadrants(left_uncropped, right_uncropped)
        nib.save(merged, paths.download / f"{case_name}_both.nii.gz")

    print_and_log(
        "[A-eye] Segmentation done - preparing results for download...",
        "info",
        LOGS_FOLDER,
    )
    sync_logs_to_output(paths.download)

    # 2. Zip folder for download
    shutil.copy2(Path("LICENSE.txt"), paths.download / "LICENSE.txt")
    zip_folder(paths.download, paths.output_zip)

    # 3. Start background thread to copy files
    increment_cases_processed()
    # send_email(user_email, "A-eye segmentation task completed successfully. You can download the results.")
    threading.Thread(
        target=copy_segmentation_data,
        args=(user_email, paths.download),
    ).start()

    # 4. add result file and metadata
    result: list = []

    for case_name in sorted(original_shapes.keys()):
        # extract_nifti_metadata provide results in a dict : [filename : metadata]
        input_metadata_dict = extract_nifti_metadata(
            str(paths.download / f"{case_name}_raw.nii.gz")
        )
        input_metadata = input_metadata_dict.get(f"{case_name}_raw.nii", {})

        result.append({
            "name": case_name,
            "input_name": f"{case_name}_raw.nii.gz",
            "left_name": f"{case_name}_left.nii.gz",
            "right_name": f"{case_name}_right.nii.gz",
            "both_name": f"{case_name}_both.nii.gz",
            "metadata": input_metadata,
        })

    return jsonify({
        "message": "Segmentation completed",
        "download_url": "/download",
        "result": result,
    }), 200


@bp.route("/profile")
@requires_auth  # Ensure only logged-in users can view this
def profile():
    user = session.get("user")
    if not user:
        return redirect(url_for("routes.login"))

    return render_template("profile.html", user=user)


@bp.route("/download", methods=["GET"])
@requires_auth
def download_files() -> Response | tuple[str, int]:
    """Download the ZIP file generated for the user logged in

    Returns:
        Response: The ZIP file
        tuple[str, int]: A "File not found" error message with HTTP 404
    """
    user_email: str = session.get("user", {}).get("email", "unknown_user")
    paths: UserPaths = get_user_paths(user_email)
    if paths.output_zip.exists():
        safe_email = user_email.replace("@", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"output_{safe_email}_{timestamp}.zip"
        return send_file(
            paths.output_zip, as_attachment=True, download_name=download_name
        )
    return "File not found", 404


@bp.route("/test-email")
@requires_auth
def test_email():
    # Get user email from session or token
    user_email = session.get("user", {}).get("email", "unknown_user")
    to = user_email  # use your real address
    body = "This is a test email from A-eye."

    try:
        msg = Message(subject="Test Email", recipients=[to], body=body)
        mail.send(msg)
        return "Email sent successfully!", 200
    except Exception as e:
        return f"Failed to send email: {e}", 500


@bp.route("/result/<filename>", methods=["GET"])
def serve_result(filename: str):
    """Serve a result file (.nii.gz) to be included in the output download

    Args:
        filename (str): name of the file to serve, in the user's download folder

    Returns:
        Response: the requested file, or a 401 JSON response if not logged in
    """
    if "user" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    user_email: str = session.get("user", {}).get("email", "unknown_user")
    file_path: Path = get_user_paths(user_email).download / filename

    return send_file(file_path, mimetype="application/gzip")


@bp.route("/result-image/<filename>", methods=["GET"])
def serve_result_image(filename: str):
    """Serve an image that is displayed in the page and will be in the output download"""
    if "user" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    user_email: str = session.get("user", {}).get("email", "unknown_user")
    file_path: Path = get_user_paths(user_email).download / filename

    return send_file(file_path)


@bp.route("/display-image/<filename>", methods=["GET"])
def serve_display_image(filename: str):
    """Serve an image that is displayed in the page"""
    if "user" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    user_email: str = session.get("user", {}).get("email", "unknown_user")
    file_path: Path = get_user_paths(user_email).visualisation / filename

    return send_file(file_path)


@bp.route("/biomarkers", methods=["POST"])
@requires_auth_api
def extract_biomarkers() -> tuple[Response, int]:
    """Run biomarker extraction for all segmented cases.

    1. loads cropped segmentation and raw images per eye
    2. runs volumetry and axial length computation
    3. saves PNG visualisations to the download folder
    4. saves a summary CSV to the download folder

    Returns:
        tuple[Response, int]: JSON response with results and HTTP status code.
            200 on success - 400 if no case names provided - 500 on failure
    """
    user_email: str = session.get("user", {}).get("email", "unknown_user")
    paths: UserPaths = get_user_paths(user_email)
    # get the body of the POST request made
    body = request.get_json(force=True) or {}
    case_names: list[str] = body.get("case_names", [])

    if not case_names:
        return jsonify({"message": "No case names provided", "status": "error"}), 400

    results = []
    csv_rows: list[dict] = []

    try:
        for case_name in case_names:
            case_result: dict = {"case_name": case_name}
            raw_path = paths.download / f"{case_name}_raw.nii.gz"

            for side, left_side in (("left", True), ("right", False)):
                eye_data, csv_row = _process_eye(
                    paths, case_name, side, left_side, raw_path
                )
                case_result[side] = eye_data
                if csv_row is not None:
                    csv_rows.append(csv_row)

            results.append(case_result)

    except Exception as error:
        current_app.logger.exception("Error during biomarker extraction")
        return jsonify({
            "message": f"Extraction failed: {error}",
            "status": "error",
        }), 500

    if csv_rows:
        csv_path = paths.download / "biomarkers.csv"
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # regenerate the ZIP to include the csv and images
    zip_folder(paths.download, paths.output_zip)

    return jsonify({
        "message": "Biomarkers extracted",
        "status": "success",
        "results": results,
    }), 200


@bp.route("/license")
def serve_license():
    return send_file(Path("LICENSE.txt"), mimetype="text/plain")


@bp.route("/reset", methods=["POST"])
@requires_auth_api
def reset_session() -> tuple[Response, int]:
    """Reset a session by :
        - cancelling any running job
        - deleting the user's uploaded files and segmentation results

    Returns:
        tuple[Response, int]: JSON response with reset status and HTTP code
    """
    user_email: str = session.get("user", {}).get("email", "unknown_user")
    paths: UserPaths = get_user_paths(user_email)

    _cancel_job(paths)

    try:
        clean_folders(user_email)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        current_app.logger.exception("Could not prepare HPC folders for upload")
        return jsonify({
            "message": "Could not reach the HPC over SSH to prepare the upload."
            "Please try again once chacha/disco SSH access is available.",
            "status": "error",
            "error": str(error),
        }), 503

    return jsonify({"message": "Session reset", "status": "success"}), 200
