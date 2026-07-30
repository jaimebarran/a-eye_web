"""Configuration of all static variables for the web platform
"""

import os
from dotenv import load_dotenv

load_dotenv()  # load env variables once

# PATHS

# configs
ALLOWED_EXTENSIONS = {"gz", "zip", "7z", "nii"}
STATS_FILE = "./data/stats.json"

# local
UPLOAD_FOLDER = "./static/upload"
AUX_BASE_FOLDER = "./nnUNet/nnUNet_inference"
AUX_INPUT_FOLDER: str = "./nnUNet/nnUNet_inference/input"
DOWNLOAD_FOLDER = "./nnUNet/nnUNet_inference/output"
OUTPUT_ZIP = "./nnUNet/nnUNet_inference/output.zip"
LOGS_FOLDER: str = "./logs"
JOBFILE_TEMPLATE = "./jobfiles/nnunet_inference_template.sh"
JOBFILE = "./jobfiles/nnunet_inference.sh"

# HPC
SSH_USER: str = "jaime.barrancohernandez@10.130.2.72"  # chacha
BASE_INPUT_HPC: str = (
    "/home/jaime.barrancohernandez/shared_datasets/nnunet/nnUNet/nnUNet_inference"
)
INPUT_HPC = (
    "/home/jaime.barrancohernandez/shared_datasets/nnunet/nnUNet/nnUNet_inference/input"
)
OUTPUT_HPC = "/home/jaime.barrancohernandez/results/nnunet"
JOBFILE_HPC = "/home/jaime.barrancohernandez/shared_datasets/nnunet/nnunet_inference.sh"

# mount
DATA_FOLDER = "/app/filer01"

# References

REF_RIGHT_AL = "data/reference_biomarkers/axial_length_nnunet_3D_N1157_right_eyes(in).csv"
REF_LEFT_AL = "data/reference_biomarkers/axial_length_nnunet_3D_N1157_left_eyes(in).csv"
REF_RIGHT_VOL= "data/reference_biomarkers/volumes_nnunet_right_eye_qc1_qc2_qc3(in).csv"
REF_LEFT_VOL = "data/reference_biomarkers/volumes_nnunet_left_eye_qc1_qc2_qc3(in).csv"
REF_METADATA = "data/reference_biomarkers/sub_metadata_non_labeled_dataset(in).csv"

# Flask configuration
class Config:
    # Flask secret key
    SECRET_KEY = os.getenv("SECRET_KEY")

    # Auth0
    AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
    AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
    AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    AUTH0_CALLBACK_URL = os.getenv("AUTH0_CALLBACK_URL")
    AUTH0_LOGOUT_URL = os.getenv("AUTH0_LOGOUT_URL")

    # Mail config
    MAIL_SERVER = os.getenv("MAIL_SERVER")
    MAIL_PORT = int(os.getenv("MAIL_PORT"))
    MAIL_USE_TLS = os.getenv("MAIL_USE_TLS")
    MAIL_USE_SSL = os.getenv("MAIL_USE_SSL")
    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER")
