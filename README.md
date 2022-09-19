# SynCT_TcMRgFUS
Official PyTorch Implementation of "Synthetic CT Skull Generation for Transcranial MR Imaging–Guided Focused Ultrasound Interventions with Conditional Adversarial Networks"

# docker: 
dockerhub: liuh26/syn_ct
how to use:
1. download Docker and NVIDIA Container Toolkit.
2. make inference with the following command:
  `docker run --gpus all --rm -v [input_directory]:/input/:ro -v [output_directory]:/output -it syn_ct`

input_directory is the directory where you put your input MR images (.nii or .nii.gz).
output_directory is the directory where you will see your output files (synthesized CT images).
