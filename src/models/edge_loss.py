import torch
import torch.nn.functional as F


def get_sobel_kernel_3d():
    """ Return a [3,1,3,3,3] sobel kernel"""

    return torch.tensor(

        [

            [[[-1, 0, 1],

              [-2, 0, 2],

              [-1, 0, 1]],

             [[-2, 0, 2],

              [-4, 0, 4],

              [-2, 0, 2]],

             [[-1, 0, 1],

              [-2, 0, 2],

              [-1, 0, 1]]],

            [[[-1, -2, -1],

              [0, 0, 0],

              [1, 2, 1]],

             [[-2, -4, -2],

              [0, 0, 0],

              [2, 4, 2]],

             [[-1, -2, -1],

              [0, 0, 0],

              [1, 2, 1]]],

            [[[-1, -2, -1],

              [-2, -4, -2],

              [-1, -2, -1]],

             [[0, 0, 0],

              [0, 0, 0],

              [0, 0, 0]],

             [[1, 2, 1],

              [2, 4, 2],

              [1, 2, 1]]]

        ]).unsqueeze(1)


def spacialGradient_3d(image):
    """ Implementation of a sobel 3d spatial gradient inspired by the kornia library.



    :param image: Tensor [B,1,H,W,D]

    :return: Tensor [B,3,H,W,D]



    :Example:

    H,W,D = (50,75,100)

    image = torch.zeros((H,W,D))

    mX,mY,mZ = torch.meshgrid(torch.arange(H),

                              torch.arange(W),

                              torch.arange(D))



    mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4

    mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)

    mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4

    mask = mask_rond & mask_carre & mask_diamand

    image[mask] = 1





    grad_image = spacialGradient_3d(image[None,None])

    """

    # sobel kernel is not implemented for 3D images yet in kornia

    # grad_image = SpatialGradient3d(mode='sobel')(image)

    kernel = get_sobel_kernel_3d().to(image.device).to(image.dtype)
    spatial_pad = [1, 1, 1, 1, 1, 1]
    image_padded = F.pad(image, spatial_pad, 'replicate').repeat(1, 3, 1, 1, 1)
    grad_image = F.conv3d(image_padded, kernel, padding=0, groups=3, stride=1)
    return grad_image


def cal_gradient_loss(input, target, mask=None):
    l1_loss = torch.nn.L1Loss()
    d1 = spacialGradient_3d(input)
    d2 = spacialGradient_3d(target)
    if mask is not None:
        d1 = d1 * mask
        d2 = d2 * mask
        loss = l1_loss(d1, d2)
        if mask.sum() != 0:
            loss = loss * mask.shape.numel() / mask.sum()
    else:
        loss = l1_loss(d1, d2)
    return loss