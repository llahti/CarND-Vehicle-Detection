import itertools
import pandas as pd

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