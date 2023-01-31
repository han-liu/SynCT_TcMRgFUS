![](https://img.shields.io/badge/Language-python-brightgreen.svg)
[![](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/han-liu/SynCT_TcMRgFUS/blob/main/LICENSE)

# SynCT_TcMRgFUS

The repository is the official PyTorch implementation of the paper:
"Synthetic CT Skull Generation for Transcranial MR Imaging–Guided Focused Ultrasound Interventions with Conditional Adversarial Networks". [[paper]](https://arxiv.org/abs/2202.10136) 

The jounal version "Evaluation of Synthetically Generated CT for use in Transcranial
Focused Ultrasound Procedures" is currently under review. The preprint version is available [here](https://arxiv.org/pdf/2210.14775.pdf), where you will find out more details in implementation and in-depth result analysis.

## Overview
We trained a 3D cGAN model to convert a `T1-weighted MRI image (input)` to a `synthetic CT image (output)`. Our approach was originally developed for Transcranial MR Imaging–Guided Focused Ultrasound Interventions (TcMRgFUS),  but can also be used as a benchmark for MR-to-CT image synthesis. Please note that our model is 3D and thus avoids the artifacts caused by the intensity inconsistency among slices (typically occur in 2D networks).

<center><img src="https://github.com/han-liu/SynCT_TcMRgFUS/blob/main/vis.png?raw=true" alt="vis" width="800"></center>

Please check out our manuscript if you are interested in the comparison between the real and synthetic CTs in (1) tFUS planning using Kranion and (2) acoustic simulation using k-wave!


If you find our repo useful to your research, please consider citing our works:

```
@article{liu2022evaluation,
  title={Evaluation of Synthetically Generated CT for use in Transcranial Focused Ultrasound Procedures},
  author={Liu, Han and Sigona, Michelle K and Manuel, Thomas J and Chen, Li Min and Dawant, Benoit M and Caskey, Charles F},
  journal={arXiv preprint arXiv:2210.14775},
  year={2022}
}

@inproceedings{liu2022synthetic,
  title={Synthetic CT skull generation for transcranial MR imaging--guided focused ultrasound interventions with conditional adversarial networks},
  author={Liu, Han and Sigona, Michelle K and Manuel, Thomas J and Chen, Li Min and Caskey, Charles F and Dawant, Benoit M},
  booktitle={Medical Imaging 2022: Image-Guided Procedures, Robotic Interventions, and Modeling},
  volume={12034},
  pages={135--143},
  year={2022},
  organization={SPIE}
}
```

If you have any questions, feel free to contact han.liu@vanderbilt.edu or open an Issue in this repo. 

## Prerequisites
* NVIDIA GPU + CUDA + cuDNN

## Installation
* create conda environment and install dependencies
```shell script
conda create --name synct python=3.8
conda activate synct
pip install -r requirements.txt
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training and Validation
```shell script
python train.py --dataroot . --model han_pix2pix --input_nc 1 --output_nc 1 --direction AtoB --netG resnet_9blocks --display_id 0 --print_freq 20 --n_epochs 3000 --n_epochs_decay 3000 --save_epoch_freq 100 --training_mode cGAN --name synct --lambda_L1 100 --lambda_Edge 10
```
As reported in the paper, the optimal performance was achieved on our validation set using hyperparameters of lambda_L1=100 and lambda_Edge=10.

## Testing
Two arguments are required (1) input_dir: this specifies where you put your input MR images (in the format of .nii or .nii.gz), and (2) output_dir: a folder to store the output synthetic CTs; this can be a new folder. Optionally, you can adjust the overlapping ratio in the sliding window inference function (default is 0.6). Higher overlapping ratio typically produces better synthetic images, but needs longer inference time. 

We provide an example MRI image [[click to download]](https://drive.google.com/file/d/1wW-MWanj74CYhpgUej0AwPPxD2h60fQq/view?usp=share_link) and our trained model [[click to download]](https://drive.google.com/file/d/1BpPVHtn5MUYQCleITXrkenhD7ZY3yYHb/view?usp=share_link). To reproduce our experimental result on the example MRI, please (1) download the model checkpoint and put it at `/src/checkpoints/jmi`, and (2) download the example MRI and put it in a folder (input_dir).

Example:
```shell script
python run_inference.py --input_dir ./AnyName --output_dir ./output --overlap_ratio 0.6
```

## Docker (use our code as out-of-box tool!) 

dockerhub: 
```shell script
liuh26/syn_ct
```

### how to use:
1. download Docker and NVIDIA Container Toolkit.
2. make inference with the following command:
```shell script
docker run --gpus all --rm -v [input_directory]:/input/:ro -v [output_directory]:/output -it syn_ct
```

where
* input_directory is the directory where you put your input MR images (.nii or .nii.gz).
* output_directory is the directory where you will see your output files (synthesized CT images).

## Acknowledgement
Our code is adapted from the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo.
