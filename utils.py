import itertools
import pandas as pd
import cv2
import numpy as np

def create_parameter_combinations(input_cspace=('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'),
                                  target_cspace=('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'),
                                  hog_orient_nbins=range(9,12),
                                  hog_pix_per_cell=range(6,10),
                                  hog_cell_per_block=range(2,4),
                                  hog_channel=(0,1,2),
                                  spatial_binning_size=((16,16), (24,24), (32,32)),
                                  hist_nbins=(10, 12, 16),
                                  hist_channels=(0, 1, 2),
                                  spatial_feat=(False, True),
                                  hist_feat=(False, True),
                                  hog_feat=(False, True)):

    combinations = itertools.product(input_cspace, target_cspace, hog_orient_nbins,
                                     hog_pix_per_cell, hog_cell_per_block, hog_channel,
                                     spatial_binning_size, hist_channels,
                                     hist_nbins,
                                     spatial_feat, hist_feat, hog_feat)
    # Define the column names which correspond to parameter names.
    column_names = ('input_cspace', 'target_cspace', 'hog_orient_nbins', 'hog_pix_per_cell', 'hog_cell_per_block',
                    'hog_channel', 'spatial_binning_size', 'hist_channels', 'hist_nbins', 'spatial_feat',
                    'hist_feat', 'hog_feat')
    df = pd.DataFrame(list(combinations), columns=column_names)
    return df


def pip(image, subimage, pos, size, border=5):
    """
    Adds sub image into image on given position and given size.
    
    :param image: Image to where subimage is placed 
    :param subimage: Image to be placed into image
    :param pos: Position of top left corner of subimage in image (x, y)
    :param size: Size of the subimage (It'll be resized to this size) (x, y)
    :param border: thickness of black border around subimage
    :return: combined image
    """

    # Coordinates of subimage
    x_left = pos[0] + border  # move subimage amount of border
    x_right = x_left + size[0]
    y_top = pos[1] + border
    y_bot = y_top + size[1]

    image[y_top - border:y_bot + border,
    x_left - border:x_right + border] = 0  # Cut black hole on left top corner
    image[y_top:y_bot, x_left:x_right] = cv2.resize(subimage, size,
                                                    interpolation=cv2.INTER_CUBIC)

    return image
