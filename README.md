![](https://img.shields.io/badge/Language-python-brightgreen.svg)
[![](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://github.com/han-liu/SynCT_TcMRgFUS/blob/main/LICENSE)

# SynCT_TcMRgFUS

The repository is the official PyTorch implementation of the [paper](https://arxiv.org/abs/2202.10136) :
"Synthetic CT Skull Generation for Transcranial MR Imaging–Guided Focused Ultrasound Interventions with Conditional Adversarial Networks".

The jounal version "Evaluation of Synthetically Generated CT for use in Transcranial
Focused Ultrasound Procedures" is currently under review. The preprint version is available [here](https://arxiv.org/pdf/2210.14775.pdf), where you will find out more details in implementation and in-depth result analysis.

<center><img src="https://github.com/han-liu/SynCT_TcMRgFUS/blob/main/vis.png?raw=true" alt="vis" width="800"></center>

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
```shell script
python run_inference.py
```
You can adjust the overlapping ratio in the sliding window inference function. Higher overlapping ratio typically produces better synthetic images, but needs longer inference time. 

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
