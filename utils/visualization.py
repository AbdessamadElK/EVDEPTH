
import numpy
from matplotlib import colors
from skimage import io
import cv2
import torch
from skimage.transform import rotate, warp

import matplotlib


def visualize_optical_flow(flow, savepath=None, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    flow = flow.transpose(1,2,0)
    flow[numpy.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = numpy.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = numpy.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = numpy.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=numpy.pi*2
    hsv[..., 0] = ang/numpy.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)
    # This all seems like an overkill, but it's just to exactly match the cv2 implementation
    bgr = numpy.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=2)


    if savepath is not None:
        out = bgr*255
        io.imsave(savepath, out.astype('uint8'))

    return bgr, (mag.min(), mag.max())


def grayscale_to_rgb(tensor, permute=False):
    # Tensor [height, width, 3], or
    # Tensor [height, width, 1], or
    # Tensor [1, height, width], or
    # Tensor [3, height, width]

    # if permute -> Convert to [height, width, 3]
    if permute:
        if tensor.size()[0] < 4:
            tensor = tensor.permute(1, 2, 0)
        if tensor.size()[2] == 1:
            return torch.stack([tensor[:, :, 0]] * 3, dim=2)
        else:
            return tensor
    else:
        if tensor.size()[0] == 1:
            return torch.stack([tensor[0, :, :]] * 3, dim=0)
        else:
            return tensor


def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points


def events_to_event_image(event_sequence, height, width, background=None, rotation_angle=None, crop_window=None,
                          horizontal_flip=False, flip_before_crop=True):
    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 1].astype(int)
    y_negative = event_sequence[~polarity, 2].astype(int)
    x_positive = event_sequence[polarity, 1].astype(int)
    y_positive = event_sequence[polarity, 2].astype(int)

    positive_histogram, _, _ = numpy.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = numpy.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = numpy.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = numpy.transpose(positive_histogram > negative_histogram)

    if background is None:
        height, width = red.shape
        background = torch.full((3, height, width), 255).byte()
    if len(background.shape) == 2:
        background = background.unsqueeze(0)
    else:
        if min(background.size()) == 1:
            background = grayscale_to_rgb(background)
        else:
            if not isinstance(background, torch.Tensor):
                background = torch.from_numpy(background)
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(numpy.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(numpy.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background

def writer_add_features(tensor_feat):
    feat_img = tensor_feat.detach().cpu().numpy()
    # img_grid = self.make_grid(feat_img)
    feat_img = numpy.sum(feat_img,axis=0)
    feat_img = feat_img -numpy.min(feat_img)
    img_grid = 255*feat_img/numpy.max(feat_img)
    img_grid = cv2.applyColorMap(numpy.array(img_grid, dtype=numpy.uint8), cv2.COLORMAP_JET)
    return img_grid



# Depth visualization
def gray_to_colormap(img, cmap='plasma'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    #img = img / (img.max() + 1e-8)

    img_ = img.flatten()
    max_value = numpy.percentile(img_, q=98)
    img = img / (max_value + 1e-8)
    
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(numpy.uint8)
    colormap[mask_invalid] = 0
    return colormap
