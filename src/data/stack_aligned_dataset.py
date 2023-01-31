import os
from glob import glob
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import nibabel as nib



class StackAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.A_paths = sorted(glob(os.path.join(self.dir_AB, 'mr') + '/*.nii.gz'))
        self.B_paths = sorted(glob(os.path.join(self.dir_AB, 'ct') + '/*.nii.gz'))
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A = self.read_img(self.A_paths[index])[0]
        B = self.read_img(self.B_paths[index])[0]
        A = torch.tensor(A).permute(2, 0, 1).float()  # 512x512x7->7x512x512
        B = torch.tensor(B).permute(2, 0, 1).float()
        B = B[B.shape[0] // 2, :, :]  # 512x512
        return {'A': A, 'B': B, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def read_img(self, img_fp):
        assert os.path.isfile(img_fp), f"{img_fp} does not exist"
        return np.asarray(nib.load(img_fp).get_fdata()), nib.load(img_fp).header
