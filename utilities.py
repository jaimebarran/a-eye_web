import os, glob, shutil, subprocess
from sys import path
# from os import getcwd
# path.append(getcwd() + "/a-eye_segmentation/3DUnet_TF1/model")
# from network import Unet_3D
# import generate_h5

# args
args_in = '/home/jaimebarranco/Desktop/test_inference/input' # origin input folder
args_out = '/home/jaimebarranco/Desktop/test_inference/output' # origin output folder
shm_size = 10 # shared memory (gb)
abs_path = '/mnt/sda1/Repos/a-eye/a-eye_segmentation/deep_learning/nnUNet/nnUNet'
rel_path = '/opt/nnunet_resources'
aux_in = f'nnUNet_inference/temp_inference/{args_in.split("/")[-1]}' # input aux folder
aux_out = f'nnUNet_inference/temp_inference/{args_out.split("/")[-1]}' # output aux folder

def main():

    # Create output local folder
    if not os.path.exists(os.path.join(abs_path, aux_out)):
        os.makedirs(os.path.join(abs_path, aux_out))
    # Copy content from origin input folder into local input folder
    copy_folder(args_in, os.path.join(abs_path, aux_in))
    # Check input filenames (need to be in nnUNet format (0000.nii.gz))
    check_filenames(os.path.join(abs_path, aux_in))

    # inference command terminal
    command = f' \
        sudo docker run \
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
    os.system(command)
    # process = subprocess.Popen(command, shell=True) # shell=True

    # Copy local output folder into origin output folder
    copy_folder(os.path.join(abs_path, aux_out), args_out)
    # Remove local folders
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
    new_file_name = f'{file_name}_0000{file_extension}' # extension for nnUNet
    os.rename(file_path, os.path.join(os.path.dirname(file_path), new_file_name))

# sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper  