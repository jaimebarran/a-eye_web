docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint nnUNet_export_model_to_zip \
  -v /home/debi/jaime/repos/a-eye/a-eye_web/nnUNet/nnUNet_trained_models:/models \
  -v "$(pwd):/out" \
  -e RESULTS_FOLDER=/models \
  jaimebarran/fw_gear_aeye:0.0.1 \
  -t 313 -o /out/A-eye_nnUNet_model_weights.zip -m 3d_fullres -tr nnUNetTrainerV2