import os
import glob
import numpy as np
from cc3d import connected_components
from dipy.align.reslice import reslice
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage import exposure
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection


def read_img(img_fp):
    assert os.path.isfile(img_fp), f"{img_fp} does not exist"
    return np.asarray(nib.load(img_fp).get_fdata()), nib.load(img_fp).header


def save_img(img, save_fp, tmp_fp=None):
    """
    :param img:
    :param save_fp:
    :param tmp_fp: str, optional.
    If provided, the saved image will have the same orientation and spacing as the template image.
    Default: saved image will be isotropic with 1mm x 1mm x 1mm
    :return:
    """
    if tmp_fp:
        img = nib.Nifti1Image(img, nib.load(tmp_fp).affine, nib.load(tmp_fp).header)
    else:
        img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, save_fp)


def reorient(img_fp, save_fp=None):
    """
    This function converts NIFTI image to nibabel RAS+ orientation
    :return: RAS+ nibabel image
    """
    nii_img = nib.load(img_fp)
    new_img = nib.as_closest_canonical(nii_img)
    nib.save(new_img, save_fp)
    return new_img


def resize(img, target_shape, order=3):
    """
    Resize image.
    :param img: np.array. image to resize.
    :param target_shape: tuple or list. shape of the resized image.
    :param order: int, optional. interpolation order: 0: nearest neighbor, 1: linear, 3: spline (default)
    :return: resized image: np.array
    """
    if img.shape == target_shape:
        return img
    return zoom(img, tuple(map(lambda x, y: x / y, target_shape, img.shape)), order=order)


def binarize(img, threshold):
    """
    Binarize a probability map with a pre-defined threshold
    """
    img[img >= threshold] = 1
    img[img != 1] = 0
    return img


def clip(img, low, high, mode='intensity'):
    """
    Clip image with low and high intensity/percentile.
    :param img: np.array. image to clip.
    :param low: int. low value for clipping.
    :param high: int. high value for clipping.
    :param mode: str, optional (default: 'intensity'). The other option is 'percentile'.
    :return: clipped image: np.array
    """
    if mode == 'percentile':
        low = np.percentile(img, low)
        high = np.percentile(img, high)
    img[img < low] = low
    img[img > high] = high
    return img


def normalize(img):
    """
    Normalize image to [0, 1]
    """
    img = img.astype('float64')
    img = img - np.min(img)
    if np.max(img) != 0:
        img /= np.max(img)
    else:
        return img
    return img


def standardize(img):
    """
    Standardize image to zero mean, unit standard deviation
    """
    img = (img - img.mean()) / img.std()
    return img


def hist_eq(img, adaptive=False):
    """
    Histogram equalization
    """
    if adaptive:
        return exposure.equalize_adapthist(img)
    return exposure.equalize_hist(img)


def get_resolution(img_fp):
    img = nib.load(img_fp)
    return img.header.get_zooms()[:3]


def isotropic(img_fp, order=1, save_fp=None, new_zooms=(1., 1., 1.)):
    """
    order: 0 nearest, 1 trilinear
    """
    img = nib.load(img_fp)
    img = nib.as_closest_canonical(img)   # optional
    data = img.get_fdata()
    affine = img.affine

    # tmp = nib.load(os.path.join('/mnt/sdb2/Research/DBS-ANT/Data', os.path.basename(img_fp)))
    # affine_correct = tmp.affine
    # zooms = tmp.header.get_zooms()[:3]

    zooms = img.header.get_zooms()[:3]
    data2, affine2 = reslice(data, affine, zooms, new_zooms, order)
    img2 = nib.Nifti1Image(data2, affine2)
    if save_fp:
        nib.save(img2, save_fp)
    return img2


def n4_correction(img_fp):
    """
    N4 bias field correction
    """
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = img_fp
    n4.inputs.bspline_fitting_distance = 300
    n4.inputs.shrink_factor = 3
    n4.inputs.n_iterations = [50, 50, 30, 20]
    n4.inputs.output_image = img_fp.replace('.nii.gz', '_corrected.nii.gz')
    n4.run()


def bias_field_correction(img_fp, save_fp=None):
    """
    N4 Bias Field Correction
    """
    print('** Start N4 Bias Field Correction **')
    img = sitk.ReadImage(img_fp)
    msk = sitk.OtsuThreshold(img, 0, 1, 200)
    img = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(img, msk)
    sitk.WriteImage(output, save_fp)
    print("** Done **")


def parse_mask(msk, labels):
    """
    Parse the unified segmentation map to multiple one-hot segmentation maps
    :param msk: np.ndarray. segmentation map with distinct labels.
    :param labels: list. list of candidate labels. E.g. [0, 1, 2, 3]
    :return: one-hot segmentation map: np.array.
    """
    segmaps = []
    for label in labels:
        segmap = np.zeros(msk.shape)
        segmap[msk == label] = 1
        segmaps.append(segmap)
    return np.array(segmaps)


def get_cc3d(mask, top=1):
    """ 26-connected neighbor
    :param mask:
    :param top: top K connected components
    :return:
    """
    msk = connected_components(mask.astype('uint8'))
    indices, counts = np.unique(msk, return_counts=True)
    indices = indices[1:]
    counts = counts[1:]
    if len(counts) >= top:
        # print(f'Found {len(counts)} connected components')
        pass
    else:
        return 'invalid'
    labels = indices[np.argpartition(counts, -top)[-top:]]
    for i in range(top):
        msk[msk == labels[i]] = 501+i
    mn = 501
    mx = 501 + top - 1
    msk[msk < mn] = 500
    msk[msk > mx] = 500
    msk = msk - 500
    return msk


def get_bbox(msk, label):
    """
    Return the bounding box of the segmentation of specific label.
    :param msk: np.array. Segmentation map.
    :param label: int. label index for bounding box.
    :return: list. coordinates of the bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    coord = np.where(msk == label)
    return [min(coord[0]), max(coord[0]), min(coord[1]), max(coord[1]), min(coord[2]), max(coord[2])]


def get_bbox_center(msk, label):
    coord = np.where(msk == label)
    cntr = [int((min(coord[0]) + max(coord[0])) / 2),
            int((min(coord[1]) + max(coord[1])) / 2),
            int((min(coord[2]) + max(coord[2])) / 2)]
    return cntr


def print_bbox(bbox, mode='range'):
    """
    :param bbox:
    :param mode:
    :return:
    """
    if mode == 'range':
        print(f"bbox: {bbox[0]}-{bbox[1]}, {bbox[2]}-{bbox[3]}, {bbox[4]}-{bbox[5]}")
    elif mode == 'dim':
        print(f"bbox: {bbox[1]-bbox[0]}, {bbox[3]-bbox[2]}, {bbox[5]-bbox[4]}")
    else:
        raise


def pad_bbox(bbox, pad):
    """
    TODO
    :param bbox:
    :param pad:
    :return:
    """
    bbox[0] = bbox[0] - pad[0]
    bbox[2] = bbox[2] - pad[1]
    bbox[4] = bbox[4] - pad[2]
    bbox[1] = bbox[1] + pad[0]
    bbox[3] = bbox[3] + pad[1]
    bbox[5] = bbox[5] + pad[2]
    return bbox


def reg_bbox(bbox, img_shape):
    """
    TODO
    :param bbox:
    :param img_shape:
    :return:
    """
    bbox[0] = max(0, bbox[0])
    bbox[2] = max(0, bbox[2])
    bbox[4] = max(0, bbox[4])
    bbox[1] = min(img_shape[0], bbox[1])
    bbox[3] = min(img_shape[1], bbox[3])
    bbox[5] = min(img_shape[2], bbox[5])
    return bbox


def get_vol_from_bbox(img, bbox):
    """

    :param img:
    :param bbox:
    :return:
    """
    return img[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1, bbox[4]:bbox[5]+1]


def parse_name(filename):
    """
    Parse a filename to (1) basename and (2) file type
    """
    parse = os.path.splitext(os.path.basename(filename))
    return parse[0], parse[1]


def reformat_time(seconds):
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def generate_label(img_dir, msk_dir, csv_fp):
    """
    Generate a csv file given two folders of paired images and masks.
    """
    f = open(csv_fp, "w")
    f.write("image,mask\n")
    print("** Generating .csv file for label **")
    img_fps = [os.path.join(img_dir, img_fp) for img_fp in glob.glob(img_dir + "/*.*")]
    msk_fps = [os.path.join(msk_dir, msk_fp) for msk_fp in glob.glob(msk_dir + "/*.*")]
    img_fps.sort()
    msk_fps.sort()
    assert len(img_fps) == len(msk_fps)
    for i in range(len(img_fps)):
        f.write(f"{img_fps[i]},{msk_fps[i]}\n")
    print(f"Successfully generated csv at {csv_fp}")


def isotropic_dir(src_dir, output_dir, order=3):
    """
    Isotropic a folder of images
    :param src_dir: str. directory of input images.
    :param output_dir: str. directory of output images.
    :param order: interpolation order. See 'resize' function.
    """
    img_fps = glob.glob(src_dir + "/*.*")
    for i, img_fp in enumerate(img_fps):
        print(f"** Isotropic: image {i + 1} **")
        basename, _ = parse_name(img_fp)
        save_fp = output_dir + "/" + basename + ".nii"
        isotropic(img_fp, order, save_fp)


def get_slice(img, ori, slc):
    """
    Extract one slice from a 3D numpy.ndarray image.
    :param img: np.array. input image to extract slice.
    :param ori: int. orientation to extract slice. 1. sagittal. 2. coronal. 3. axial.
    :param slc: int. slice number to extract.
    :return: extracted 2D slice: np.array.
    """
    if ori == 1 and 0 <= slc < img.shape[0]:
        return img[slc, :, :]
    elif ori == 2 and 0 <= slc < img.shape[1]:
        return img[:, slc, :]
    elif ori == 3 and 0 <= slc < img.shape[2]:
        return img[:, :, slc]
    else:
        raise Exception('Invalid orientation or slice number')
