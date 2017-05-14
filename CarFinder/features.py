"""
Code from udacity lesson
https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/40ac880a-7ccc-4145-a864-6b0b99ea31e9

Modified to have object oriented touch.
"""

import numpy as np
import cv2
from skimage.feature import hog


class Features:
    def __init__(self):
        # Defines color space conversion, if not defined then conversion
        # is not done
        self.color_space = cv2.COLOR_BGR2LUV

        # Defines size of spatial color binning
        self.spatial_binning_size = (32, 32)

        # Color histogram parameters
        self.hist_nbins = 32  # Number of bins
        self.hist_bins_range = (0, 256)  # Range of bins

        # HOG parameters
        self.hog_channel = 'ALL'  # Choices: 0, 1, 2, 'ALL'
        self.orient = 6  # Number of orientation bins
        self.pix_per_cell = (16, 16)  # Pixels per HOG-cell
        self.cell_per_block = (2, 2)  # cells per HOG-block

        # Define what feature vectors to use
        self.spatial_feat = True
        self.hist_feat = True
        self.hog_feat = True

    def color_hist(self, img):
        """
        This function computes color histogram features for each color channel 
        and then concatenates result and return it.

        :param img: Feature image
        :return: Concatenated histogram of all 3 color channels
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_nbins,
                                     range=self.hist_bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_nbins,
                                     range=self.hist_bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_nbins,
                                     range=self.hist_bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate(
            (channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def get_hog_features(self, img, vis=False, feature_vec=True):
        """
        Define a function to return HOG features and visualization
        
        :param img: Feature image
        :param vis: if True, generate HOG image visualization
        :param feature_vec: if True, generate feature vector 
        :return: 
        """
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=self.pix_per_cell,
                                      cells_per_block=self.cell_per_block,
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec,
                                      block_norm='L2-Hys')
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                           pixels_per_cell=self.pix_per_cell,
                           cells_per_block=self.cell_per_block,
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec,
                           block_norm='L2-Hys')
            return features

    def single_img_features(self, img):
        """
        Define a function to extract features from a single image window
        This function is very similar to extract_features()
        just for a single image rather than list of images
        
        :param img: Feature image
        :return: Feature vector
        """
        # 1) Define an empty list to receive features
        img_features = []

        # Convert color space
        if self.color_space:
            feature_image = cv2.cvtColor(img, self.color_space)
        else:
            feature_image = np.copy(img)

        # 3) Compute spatial features if flag is set
        if self.spatial_feat:
            spatial_features = cv2.resize(img, self.spatial_binning_size).ravel()
            # 4) Append features to list
            img_features.append(spatial_features)

        # 5) Compute histogram features if flag is set
        if self.hist_feat:
            hist_features = self.color_hist(feature_image)

            # 6) Append features to list
            img_features.append(hist_features)

        # 7) Compute HOG features if flag is set
        if self.hog_feat:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(
                        self.get_hog_features(feature_image[:, :, channel],
                                         vis=False, feature_vec=True))
            else:
                hog_features = self.get_hog_features(
                    feature_image[:, :, self.hog_channel],
                    vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)



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
    # Ensure that features are float64
    X = np.array(features).astype(np.float64)
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