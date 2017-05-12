import itertools
import pandas as pd

def create_parameter_combinations(color_space=('RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'),
                                  orient=range(9,12),
                                  pix_per_cell=range(6,10),
                                  cell_per_block=range(2,4),
                                  hog_channel=(0,1,2,'ALL'),
                                  spatial_size=((16,16), (24,24), (32,32)),
                                  hist_bins=(10, 12, 16),
                                  spatial_feat=(False, True),
                                  hist_feat=(False, True),
                                  hog_feat=(False, True)):

    combinations = itertools.product(color_space, orient, pix_per_cell,
                                     cell_per_block, hog_channel, spatial_size,
                                     hist_bins, spatial_feat, hist_feat,
                                     hog_feat)
    # Define the column names which correspond to parameter names.
    column_names = ('color_space', 'orient', 'pix_per_cell', 'cell_per_block',
                    'hog_channel', 'spatial_size', 'hist_bins', 'spatial_feat',
                    'hist_feat', 'hog_feat')
    df = pd.DataFrame(list(combinations), columns=column_names)
    return df