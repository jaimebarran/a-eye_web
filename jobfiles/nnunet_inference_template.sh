#!/bin/bash
#SBATCH --job-name=nnUNetv1_inf  # job name
#SBATCH --partition=Dance         # cluster partition
#SBATCH --account=mattech         # Slurm account/association
#SBATCH --qos=normal              # avoid jobs landing with QOS=(null)
#SBATCH --nodes=1                # number of nodes to use
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # moderate CPU headroom for multi-image preprocessing and inference
#SBATCH --mem=16gb               # scaled from the 3.65 GB single-image run with room for batch variation
#SBATCH --time=04:00:00          # sized for batches of normal images instead of a single ~1 min case
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=/home/jaime.barrancohernandez/results/nnunet/nnUNet_predict.%N.%j.%a.out  # output log
#SBATCH --error=/home/jaime.barrancohernandez/results/nnunet/nnUNet_predict.%N.%j.%a.err   # error log
#SBATCH --mail-type=BEGIN,END  # send email when job begins and ends
#SBATCH --mail-user=jaime.barrancohernandez@hevs.ch # email address to send notifications. Replace with $MAIL_USER

apptainer exec \
    --nv \
    --bind /home/jaime.barrancohernandez/shared_datasets/nnunet/nnUNet:/opt/nnunet_resources \
    --bind /home/jaime.barrancohernandez/shared_datasets/nnunet/nnUNet/nnUNet_inference/input:/input \
    --bind /home/jaime.barrancohernandez/results/nnunet:/output \
    /home/jaime.barrancohernandez/shared_datasets/nnunet/nnunet.sif \
    nnUNet_predict \
    -i /input \
    -o /output \
    -tr nnUNetTrainerV2 \
    -m 3d_fullres \
    -p nnUNetPlansv2.1 \
    -t Task313_Eye
