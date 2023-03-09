import os, glob, shutil, subprocess
import dicom2nifti
import argparse
import logging

enable_args = True

if enable_args:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input folder')
    parser.add_argument('-o', '--output', type=str, required=True, help='output folder')
    args = parser.parse_args()
    args_in = args.input
    args_out = args.output
else:
    args_in = '/home/jaimebarranco/Desktop/test_inference/input' # origin input folder
    args_out = '/home/jaimebarranco/Desktop/test_inference/output' # origin output folder

shm_size = 10 # shared memory (gb)
abs_path = '/mnt/sda1/Repos/a-eye/a-eye_segmentation/deep_learning/nnUNet/nnUNet'
rel_path = '/opt/nnunet_resources'
aux_in = f'nnUNet_inference/temp_inference/{args_in.split("/")[-1]}' # input aux folder
aux_out = f'nnUNet_inference/temp_inference/{args_out.split("/")[-1]}' # output aux folder

def main():

    # Create output aux folder
    if not os.path.exists(os.path.join(abs_path, aux_out)):
        os.makedirs(os.path.join(abs_path, aux_out))
    # Copy content from origin input folder into local input folder
    copy_folder(args_in, os.path.join(abs_path, aux_in))

    # Convert to nifti
    convert_to_nifti(os.path.join(abs_path, aux_in))

    # Check input filenames (need to be in nnUNet format (0000.nii.gz))
    check_filenames(os.path.join(abs_path, aux_in))

    # inference command terminal
    command = f' \
        echo $SUDO_PWD | sudo -S -s \
        docker run --rm \
        --gpus device=0 \
        --shm-size={shm_size}gb \
        -v {abs_path}:{rel_path} \
        petermcgor/nnunet:0.0.1 nnUNet_predict \
        -i {rel_path}/{aux_in} \
        -o {rel_path}/{aux_out} \
        -tr nnUNetTrainerV2 \
        -ctr nnUNetTrainerV2CascadeFullRes \
        -m 3d_fullres \
        -p nnUNetPlansv2.1 \
        -t Task313_Eye \
    '
    # print(command)
    logging.info(f'AEye: nnUNet inference command: \n{command}')
    os.system(command)

    # Copy aux output folder into origin output folder
    copy_folder(os.path.join(abs_path, aux_out), args_out)
    # Remove aux folders
    delete_folder(os.path.join(abs_path, aux_in))
    delete_folder(os.path.join(abs_path, aux_out))
    
    return '\nDone! \n'

def copy_folder(source, destination):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

def delete_folder(folder):
    shutil.rmtree(folder)

def check_filenames(folder):
    file_paths = glob.glob(f'{folder}/*.nii.gz')
    for file_path in file_paths:
        file_extensions = []
        while True:
            file_path, ext = os.path.splitext(file_path)
            if ext:
                file_extensions.insert(0, ext)
            else:
                break
        file_extension = ''.join(file_extensions)
        file_name = os.path.basename(file_path)
        file_path = f'{file_path}{file_extension}'
        # print(f'file name: {file_name}')
        # print(f'file extension: {file_extension}')
        # print(f'absolute file path: {file_path}')
        if not str(file_name).endswith('_0000'):
            correct_filename(file_path, file_name, file_extension)

def correct_filename(file_path, file_name, file_extension):
    # print('Changing filename')
    logging.info('AEYE: Changing filename to nnUNet format...')
    new_file_name = f'{file_name}_0000{file_extension}' # extension for nnUNet
    os.rename(file_path, os.path.join(os.path.dirname(file_path), new_file_name))

def convert_to_nifti(folder):
    # Get a list of all DICOM folders in the input folder
    dcm_folders = sorted([f.path for f in os.scandir(folder) if f.is_dir() and f.name != '.DS_Store'])
    if len(dcm_folders) >0:
        logging.info('AEye: Converting DICOM to NIfTI format...')
        # Convert each DICOM folder to NIfTI format
        for dcm_folder in dcm_folders:
            filename = str(os.path.basename(dcm_folder) + '.nii.gz')
            dicom2nifti.dicom_series_to_nifti(dcm_folder, f'{folder}/{filename}', reorient_nifti=False)
            # cmd = ["dcm2niix", "-f", filename, "-z", "y", "-o", output_nifti_folder, input_dicom_folder]
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
            # _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()