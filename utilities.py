import os, glob
from sys import path
# from os import getcwd
# path.append(getcwd() + "/a-eye_segmentation/3DUnet_TF1/model")
# from network import Unet_3D
# import generate_h5

# args
shm_size = 10 # shared memory (gb)
abs_path = '/mnt/sda1/Repos/a-eye/a-eye_segmentation/deep_learning/nnUNet/nnUNet'
rel_path = '/opt/nnunet_resources'
args_i = 'nnUNet_inference/test/input1' # input folder
args_o = 'nnUNet_inference/test/output1' # output folder

def main():

    check_filenames(os.path.join(abs_path, args_i))

    # inference command terminal
    command = f' \
        sudo docker run \
        --gpus device=0 \
        --shm-size={shm_size}gb \
        -v {abs_path}:{rel_path} \
        petermcgor/nnunet:0.0.1 nnUNet_predict \
        -i {rel_path}/{args_i} \
        -o {rel_path}/{args_o} \
        -tr nnUNetTrainerV2 \
        -ctr nnUNetTrainerV2CascadeFullRes \
        -m 3d_fullres \
        -p nnUNetPlansv2.1 \
        -t Task313_Eye \
    '
    print(command)
    os.system(command)

    return 'Running nnUNet inference...'

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
        print(f'file name: {file_name}')
        print(f'file extension: {file_extension}')
        print(f'absolute file path: {file_path}')
        if not str(file_name).endswith('_0000'):
            correct_filename(file_path, file_name, file_extension)

def correct_filename(file_path, file_name, file_extension):
    print('Changing filename')
    new_file_name = f'{file_name}_0000{file_extension}' # extension for nnUNet
    os.rename(file_path, os.path.join(os.path.dirname(file_path), new_file_name))

# sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper  