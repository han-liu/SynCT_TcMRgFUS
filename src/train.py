import time
from options.train_options import TrainOptions
from models import create_model
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    LoadImaged,
    RandSpatialCropd,
    RandAdjustContrastd,
    CropForegroundd,
    RandZoomd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandShiftIntensityd
)
from util.visualizer import Visualizer
from glob import glob
import numpy as np


if __name__ == '__main__':
    mr_paths = sorted(glob('./bone_data/3D/mr' + '/*.nii'))
    ct_paths = sorted(glob('./bone_data/3D/processed_ct' + '/*.nii'))

    data_dicts = [{"A": mr_path, "B": ct_path, 'A_paths': mr_path, 'B_paths': ct_path} for mr_path, ct_path in zip(mr_paths, ct_paths)]
    train_files = data_dicts[:66]
    trainTransform = Compose([
        LoadImaged(keys=["A", "B"]),
        AddChanneld(keys=["A", "B"]),
        
        # MRI pre-processing
        NormalizeIntensityd(keys="A", nonzero=True),  # z-score normalization
        ScaleIntensityRangePercentilesd(keys="A", lower=0.01, upper=99.9, b_min=-1.0, b_max=1.0, clip=True, relative=False), # normalize the intensity to [-1, 1]
        
        # CT pre-processing
        ScaleIntensityRanged(keys="B", a_min=-1024, a_max=3071, b_min=-1.0, b_max=1.0, clip=True),
        
        # Spatial augmentation
        # RandAffined(keys=["A", "B"],
        #             prob=0.2,
        #             rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        #             mode=('bilinear', 'bilinear')
        #             ),

        # crop 256 x 256 x 32 volumes
        RandSpatialCropd(keys=["A", "B"], roi_size=(256, 256, 32), random_size=False),  # randomly crop patches of 256 x 256 x 32

        # Intensity augmentation
        RandShiftIntensityd(keys="A", offsets=(-0.1, 0.1), prob=0.2),
        RandAdjustContrastd(keys="A", prob=0.2, gamma=(0.8, 1.2)),
        ToTensord(keys=["A", "B"])])

    train_ds = Dataset(data=train_files, transform=trainTransform)
    train_loader = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4)

    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_loader)    # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
