import os
import os.path as osp
import ants
from models import create_model
from PIL import Image
import numpy as np
import torch
from utility import *
from glob import glob
from time import time
from utility import *
import skimage.morphology


GT_DIR = '/mnt/sdb2/Research/Bone/results/SPIE/2D/real'

if __name__ == '__main__':

    img_path = '/mnt/sdb2/Research/Bone/results/JMI/ResNet_cus/3566_fake_CT.nii.gz'
    ct = ants.image_read(img_path)
    info = [ct.origin, ct.spacing, ct.direction]
    mask = ct.numpy().copy()
    mask[mask > 400] = 1  # binary thresholding
    mask[mask != 1] = 0
    mask = get_cc3d(mask, top=1)  # largest connected component
    struct = skimage.morphology.ball(radius=4)
    mask = skimage.morphology.binary_dilation(mask, struct).astype('uint8')

    # process GT
    # gt = ants.image_read(osp.join(GT_DIR, osp.basename(img_path).replace('fake', 'real')))
    # gt_mask = gt.numpy().copy()
    # gt_mask[gt_mask > 400] = 1  # binary thresholding
    # gt_mask[gt_mask != 1] = 0
    # gt_mask = get_cc3d(gt_mask, top=1)  # largest connected component
    # struct = skimage.morphology.ball(radius=4)
    # gt_mask = skimage.morphology.binary_dilation(gt_mask, struct).astype('uint8')

    output = ct.numpy() * mask
    output = ants.from_numpy(output, origin=info[0], spacing=info[1], direction=info[2])
    ants.image_write(output, osp.join('/media/liuhan/HanLiu/Bone/masks', osp.basename(img_path)))

