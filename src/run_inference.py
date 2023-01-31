import os
import os.path as osp
import ants
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
import numpy as np
import torch
from time import time
from tqdm import tqdm
from utility import *
from glob import glob
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AddChanneld,
    Compose,
    ScaleIntensityd,
    ToTensord,
    LoadImaged,
    RandSpatialCropd,
    RandAdjustContrastd,
    CropForegroundd,
    RandZoomd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
)


def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def load_model(opt):
    opt.num_threads = 0  
    opt.batch_size = 1  
    opt.serial_batches = True  
    opt.no_flip = True  
    model = create_model(opt) 
    model.setup(opt)  
    if opt.eval:
        model.eval()
    return model


def mr_to_ct(
    img_fp: str,
    model: torch.nn.Module, 
    transform, 
    info: list, 
    save_fp: str=None,
    overlap_ratio: float=0.6,
    ) -> None:

    data = transform({'A': img_fp})["A"]
    start = time()

    print('Performing sliding window inference.....')
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=data.unsqueeze(0),
            roi_size=(256, 256, 32),
            sw_batch_size=4,
            predictor=model,
            overlap=overlap_ratio)  

    output = output.squeeze(0).squeeze(0).cpu().detach().numpy()

    print('Done...')
    print(f'Time elapsed: {time()-start:.3f} seconds')
    output = output * 2047.5 + 1023.5  # map to CT hounsfield units
    ants.image_write(ants.from_numpy(output, origin=info[0], spacing=info[1], direction=info[2]), save_fp)
    return output


if __name__ == '__main__':

    opt = TestOptions().parse()

    # override the arguments for our trained model
    opt.model = 'han_pix2pix'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.direction = 'AtoB'
    opt.netG = 'resnet_9blocks'
    opt.name = 'jmi'
    opt.epoch = 'best'
    model = load_model(opt).netG

    # check input and output directories
    mkdir(opt.output_dir)
    assert osp.exists(opt.input_dir)
    mr_paths = sorted(glob(opt.input_dir + '/*'))
    assert len(mr_paths) > 0, 'At least one input image is required.'

    # input data transform
    transform = Compose([
        LoadImaged(keys="A"),
        AddChanneld(keys="A"),
        NormalizeIntensityd(keys="A", nonzero=True),
        ScaleIntensityRangePercentilesd(keys="A", lower=0.01, upper=99.9, b_min=-1.0, b_max=1.0, clip=True, relative=False),
        ToTensord(keys="A"),
        ])

    for mr_path in mr_paths:
        pid = os.path.basename(mr_path).split('.')[0]
        mr = ants.image_read(mr_path)
        info = [mr.origin, mr.spacing, mr.direction]
        output_path = os.path.join(opt.output_dir, f'{pid}_sCT.nii.gz')
        f_ct = mr_to_ct(mr_path, model, transform, info, output_path, opt.overlap_ratio)

   