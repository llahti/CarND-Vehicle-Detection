"""
Code from udacity lesson
https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/40ac880a-7ccc-4145-a864-6b0b99ea31e9

Modified to have object oriented touch.
"""

import numpy as np
import cv2
from skimage.feature import hog


class Features:
    """
    This class is a abstraction of feature extraction. 
    
    Constructor default parameters are known best parameters. 
    """

    # Below ranges might be useful when this class is converted to accept float
    # datatype. Currently used uint8 implementation is not using this.
    # histogram_ranges = (('RGB', (0, 1), (0, 1), (0, 1)),
    #                     ('LAB', (0, 100), (-127, 127), (-127, 127)),
    #                     ('HLS', (0, 360), (0, 1), (0,1)),
    #                     ('HSV', (0, 360), (0, 1), (0, 1)),
    #                     ('YCrCb', (0, 255), (0, 255), (0, 255)), # specifically for YCrCb444, but should work for 422 and 420
    #                     ('LUV', (0, 100), (-134, 220), (-140, 122))
    #                     )

    # Supported colorspaces
    supported_color_spaces = ('BGR', 'HLS', 'HSV', 'LAB', 'LUV', 'RGB', 'YCrCb')

    # This dictionary is mapping of input and target colorspaces to cv2.COLOR_*
    # colorspace conversion constants.
    # Format is:  colorspace_transformations[input][output]
    colorspace_transformations = {'RGB': {'RGB': None,
                                          'HSV': cv2.COLOR_RGB2HSV,
                                          'HLS': cv2.COLOR_RGB2HLS,
                                          'LAB': cv2.COLOR_RGB2LAB,
                                          'LUV': cv2.COLOR_RGB2LUV,
                                          'BGR': cv2.COLOR_RGB2BGR,
                                          'YCrCb': cv2.COLOR_RGB2YCrCb},
                                  'BGR': {'RGB': cv2.COLOR_BGR2RGB,
                                          'HSV': cv2.COLOR_BGR2HSV,
                                          'HLS': cv2.COLOR_BGR2HLS,
                                          'LAB': cv2.COLOR_BGR2LAB,
                                          'LUV': cv2.COLOR_BGR2LUV,
                                          'BGR': None,
                                          'YCrCb': cv2.COLOR_BGR2YCrCb},
                                  'HSV': {'RGB': cv2.COLOR_HSV2RGB,
                                          'HSV': None,
                                          'HLS': None,
                                          'LAB': None,
                                          'LUV': None,
                                          'BGR': cv2.COLOR_HSV2BGR,
                                          'YCrCb': None},
                                  'HLS': {'RGB': cv2.COLOR_HLS2RGB,
                                          'HSV': None,
                                          'HLS': None,
                                          'LAB': None,
                                          'LUV': None,
                                          'BGR': cv2.COLOR_HLS2BGR,
                                          'YCrCb': None},
                                  'LAB': {'RGB': cv2.COLOR_LAB2RGB,
                                          'HSV': None,
                                          'HLS': None,
                                          'LAB': None,
                                          'LUV': None,
                                          'BGR': cv2.COLOR_LAB2BGR,
                                          'YCrCb': None},
                                  'LUV': {'RGB': cv2.COLOR_LUV2RGB,
                                          'HSV': None,
                                          'HLS': None,
                                          'LAB': None,
                                          'LUV': None,
                                          'BGR': cv2.COLOR_LUV2BGR,
                                          'YCrCb': None},
                                  'YCrCb': {'RGB': cv2.COLOR_YCrCb2RGB,
                                            'HSV': None,
                                            'HLS': None,
                                            'LAB': None,
                                            'LUV': None,
                                            'BGR': cv2.COLOR_YCrCb2BGR,
                                            'YCrCb': None},
                                  }

    def __init__(self, input_cspace='BGR',
                 target_cspace='YCrCb',
                 spatial_binning_size=(15, 15),
                 hist_nbins=256,
                 hist_channels=(0, 1, 2),
                 hog_channel=( 1, ),
                 hog_orient_nbins=10,
                 hog_pix_per_cell=(14, 14),
                 hog_cell_per_block=(2, 2),
                 spatial_feat=True,
                 hist_feat=True,
                 hog_feat=True):

        if not Features.is_color_space_valid(input_cspace):
            raise Exception(
                "Target colorspace {} is not supported".format(target_cspace))
        if not Features.is_color_space_valid(target_cspace):
            raise Exception(
                "Input colorspace {} is not supported".format(input_cspace))

        # Defines color space conversion, if not defined then conversion
        # is not done
        self.color_space_trans = self.get_cspace_transformation(input_cspace, target_cspace)
        self.input_cspace = input_cspace
        self.target_cspace = target_cspace

        # Defines size of spatial color binning
        self.spatial_binning_size = spatial_binning_size

        # Color histogram parameters
        self.hist_nbins = hist_nbins  # Number of bins
        self.hist_bins_range = (0, 256)  # Range of bins
        self.hist_channels = hist_channels   # Defines which channels are used to take histogram

        # HOG parameters
        self.hog_channel = hog_channel # Choices: 0, 1, 2
        self.hog_orient_nbins = hog_orient_nbins  # Number of orientation bins
        self.hog_pix_per_cell = hog_pix_per_cell  # Pixels per HOG-cell
        self.hog_cell_per_block = hog_cell_per_block  # cells per HOG-block

        # Define what feature vectors to use
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

    def bin_spatial(self, img):
        """
        This method creates a spatially binned feature vector.
        1. It resizes image to size defined by self.spatial_binning_size
        2. Then convert 2D image to 1D feature vector
        
        :param img: uint8 
        :return: uint8 1D feature vector
        """
        #assert img.dtype == np.uint8, "Only np.uint8 is supported"
        # Get resized version of image
        spatial_features = cv2.resize(img, self.spatial_binning_size)
        # Make it a 1D-vector and return it
        return spatial_features.ravel()

    def color_hist(self, img):
        """
        This method computes color histogram of each wanted colorplane. These 
        are defined in self.hist_channels.
        
        Then feature vector is generated by combining individual histograms into
        1D feature vector-
        

        :param img: Feature image, uint8
        :return: Concatenated histogram of all 3 color channels. dtype=uint8
        """
        #assert img.dtype == np.uint8, "Only np.uint8 is supported"
        # Compute the histogram of the color channels separately
        features = []
        for i in self.hist_channels:
            hist = np.histogram(img[:, :, i], bins=self.hist_nbins,)
                                #range=(0, 255))
                                #range=self.hist_range[i+1])
            # Append only the histogram part
            features.append(hist[0])

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate(features)
        # Return the individual histograms as a 1D feature vector
        return hist_features

    def convert_colorspace(self, img):
        """
        
        :param img: image in uint8 format 
        :return: 
        """
        assert img.dtype == np.uint8, "Only np.uint8 is supported"
        # Convert color space
        if self.color_space_trans:
            feature_image = cv2.cvtColor(img, self.color_space_trans)
        else:
            feature_image = np.copy(img)
        return feature_image

    @staticmethod
    def get_cspace_transformation(input_cspace, target_cspace):
        """
        This method returns cv2 colorspace transformation constant for given 
        input and output colorspaces
        
        Warning! Currently colorspace transformation is working reliably when 
        atleast one colorspace is RGB or BGR. This is due to the fact that
        openCV does not provide conversions between e.g. HSV and LUV. In those 
        cases it is better to do conversion in following way e.g. HSV-->BGR-->LUV
        
        :param input_cspace: choise('BGR', 'HLS', 'HSV', 'LAB', 'LUV', 'RGB', 'YCrCb')
        :param target_cspace: choise('BGR', 'HLS', 'HSV', 'LAB', 'LUV', 'RGB', 'YCrCb')
        :return: cv2.COLOR_* constant to define colorspace transformation.
        """
        if target_cspace not in Features.supported_color_spaces:
            raise Exception(
                "Target colorspace {} is not supported".format(target_cspace))
        if input_cspace not in Features.supported_color_spaces:
            raise Exception(
                "Input colorspace {} is not supported".format(input_cspace))

        # Format is colorspace_transformations[input][output]
        cspace_trans = Features.colorspace_transformations[input_cspace][
            target_cspace]
        return cspace_trans

    def get_hog_features(self, img, vis=False, feature_vec=True):
        """
        This method return HOG features and visualization.
        
        :param img: 1-channel uint8 image
        :param vis: if True, generate HOG image visualization
        :param feature_vec: if True, generate feature vector 
        :return: Returns feature vector and/or visualization depending of the 
        vis and feat_vec parameters.
        """
        #assert img.dtype == np.uint8, "Only np.uint8 is supported"
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self.hog_orient_nbins,
                                      pixels_per_cell=self.hog_pix_per_cell,
                                      cells_per_block=self.hog_cell_per_block,
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec,
                                      block_norm='L2-Hys')
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.hog_orient_nbins,
                           pixels_per_cell=self.hog_pix_per_cell,
                           cells_per_block=self.hog_cell_per_block,
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec,
                           block_norm='L2-Hys')
            return features

    @staticmethod
    def is_color_space_valid(cspace):
        """
        This method check is given colorspace string valid. In other words is 
        it in supported_colorspaces.
        
        :param cspace: 
        :return: True if valid, False if not valid
        """
        return cspace in Features.supported_color_spaces

    def single_img_features(self, img):
        """
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        
        :param img: Feature image
        :return: Feature vector
        """

        # Check that image has 3 channels
        assert img.shape[2] == 3, "Channel count is not 3"
        # 1) Define an empty list to receive features
        img_features = []

        #img = img.astype(dtype=np.float32)
        feature_image = self.convert_colorspace(img)
        feature_image = feature_image.astype(dtype=np.float32)/255
        feature_image
        img = feature_image

        # 3) Compute spatial features if flag is set
        if self.spatial_feat:
            spatial_features = self.bin_spatial(img)
            # 4) Append features to list
            img_features.append(spatial_features)


        # 5) Compute histogram features if flag is set
        if self.hist_feat:
            hist_features = self.color_hist(feature_image)

            # 6) Append features to list
            img_features.append(hist_features)


        # 7) Compute HOG features if flag is set
        if self.hog_feat:
            hog_features = []
            for ch in self.hog_channel:
                feat = self.get_hog_features(feature_image[:, :, ch],
                                          vis=False, feature_vec=True)
                hog_features.extend(feat)
            # if self.hog_channel == 'ALL':
            #     hog_features = []
            #     for channel in range(feature_image.shape[2]):
            #         hog_features.extend(
            #             self.get_hog_features(feature_image[:, :, channel],
            #                              vis=False, feature_vec=True))
            # else:
            #     hog_features = self.get_hog_features(
            #         feature_image[:, :, self.hog_channel],
            #         vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        feat = np.concatenate(img_features)
        return feat



    def extract_features(self, imgs):
        """
        Define a function to extract features from a list of images
        Have this function call bin_spatial() and color_hist()
        
        :param imgs: 
        :return: 
        """
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for img in imgs:
            feat = self.single_img_features(img)
            features.append(feat)
        # Return list of feature vectors
        return np.array(features)

if __name__ == "__main__":
    import CarFinder.utils as utils
    import glob
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    images, labels = utils.load_test_images()

    f = Features()

    print("Extracting features... ", end='')
    features = f.extract_features(images)

    # Ensure that features are float64. Standardscaler only accepts float64 type.
    X = features.astype(dtype=np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    print("OK!")

    print("Cross validating...", end='')
    # Split up data into randomized training and test sets
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,
                                 random_state=888)

    # Use a linear SVC
    svc = SVC(kernel='rbf', C=100, gamma=0.0005)

    # Calculate cross valitated score (this is more reliable way)
    score = cross_val_score(svc, scaled_X, labels, cv=sss, n_jobs=1).mean()
    print("OK!")

    print("Cross validation score is {}".format(score))