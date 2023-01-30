![](https://img.shields.io/badge/Language-python-brightgreen.svg)
[![](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/han-liu/SynCT_TcMRgFUS/blob/main/LICENSE)

# SynCT_TcMRgFUS

The repository is the official PyTorch implementation of the paper:
"Synthetic CT Skull Generation for Transcranial MR Imaging–Guided Focused Ultrasound Interventions with Conditional Adversarial Networks"
The paper link is https://arxiv.org/abs/2202.10136

Our jounal version is currently under review and the pre-print version "Evaluation of Synthetically Generated CT for use in Transcranial
Focused Ultrasound Procedures" is available at https://arxiv.org/pdf/2210.14775.pdf, where you will find out more details in implementation and in-depth result analysis.

If you use this code, please cite our work, the reference is
```
@inproceedings{liu2022synthetic,
  title={Synthetic CT skull generation for transcranial MR imaging--guided focused ultrasound interventions with conditional adversarial networks},
  author={Liu, Han and Sigona, Michelle K and Manuel, Thomas J and Chen, Li Min and Caskey, Charles F and Dawant, Benoit M},
  booktitle={Medical Imaging 2022: Image-Guided Procedures, Robotic Interventions, and Modeling},
  volume={12034},
  pages={135--143},
  year={2022},
  organization={SPIE}
}

@article{liu2022evaluation,
  title={Evaluation of Synthetically Generated CT for use in Transcranial Focused Ultrasound Procedures},
  author={Liu, Han and Sigona, Michelle K and Manuel, Thomas J and Chen, Li Min and Dawant, Benoit M and Caskey, Charles F},
  journal={arXiv preprint arXiv:2210.14775},
  year={2022}
}
```

If you have any problem, feel free to contact han.liu@vanderbilt.edu or open an Issue in this repo. 

## Prerequisites
* NVIDIA GPU + CUDA + cuDNN

## Installation
We suggest installing the dependencies using Anaconda
* create the environment
```shell script
conda create --name synct python=3.8
```
* Install PyTorch and other dependencies:
```shell script
conda activate synct
pip install -r requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
## Training and Validation
```shell script
conda activate synct
python -W ignore train.py --dataroot . --model han_pix2pix --input_nc 1 --output_nc 1 --direction AtoB --netG resnet_9blocks --display_id 0 --print_freq 20 --n_epochs 3000 --n_epochs_decay 3000 --save_epoch_freq 100 --training_mode cGAN --name synct --lambda_L1 100 --lambda_Edge 10
```
As reported in the paper, we found that the optimal performance was achieved on our validation set using hyperparameters of lambda_L1=100 and lambda_Edge 10.

## Testing

In the testing phase, you can create a new folder for the test dataset. 
For example, if you want to use the test set of ISBI challenge, create a new subfolder ./challenge
under sample\_dataset (parallel to train, val, test).
The naming of files does not need to follow the strict pattern described in Dataset and Data Conversion, 
but the files should end with {MODALITY(MASK)}.{SUFFIX}. The reference mask files are not needed.
If the reference mask files are provided, the segmentation metrics will be output.

Remember to switch to Testing mode in 'models/ms_model.py' file. This needs to be done manually for now and will be cleaned up later.

Example:
```shell script
conda activate DL
python test.py --load_str val0test4 --input_nc 3 --testSize 512 --name experiment_name --epoch 300 --phase test --eval
```

## docker (use our code as out-of-box tool) 

dockerhub: liuh26/syn_ct
how to use:
1. download Docker and NVIDIA Container Toolkit.
2. make inference with the following command:
  `docker run --gpus all --rm -v [input_directory]:/input/:ro -v [output_directory]:/output -it syn_ct`

where
* input_directory is the directory where you put your input MR images (.nii or .nii.gz).
* output_directory is the directory where you will see your output files (synthesized CT images).
