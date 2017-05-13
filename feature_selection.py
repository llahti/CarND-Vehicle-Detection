# This code is based on udacity example
# https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/40ac880a-7ccc-4145-a864-6b0b99ea31e9


import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from lesson_functions import *
from sklearn.model_selection import train_test_split
from dataset import *
from utils import *
import pandas as pd
import tqdm


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(
                    get_hog_features(feature_image[:, :, channel],
                                     orient, pix_per_cell, cell_per_block,
                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


save_results = True
filename = './model_selection/feature_results_temp.csv'


images, y = load_full()

# Get different parameter combinations
# Tweak these parameters and see how the results change.
# It is advisable to try out only with few different parameter combinations
# to save time and get an idea what are parameter ranges should be fine tuned
#

combinations = create_parameter_combinations(color_space=['LUV', 'YCrCb'], orient=[4, 5, 6], pix_per_cell=[13, 14, 15],
                                             cell_per_block=[2, 3, 4], spatial_size=((14,14), (15,15), (16,16), ),
                                             hist_bins=[256, ], hog_channel=[0, 1, 2, 'ALL'], hog_feat=[True, ], hist_feat=[True,], spatial_feat=[True,])



# Create result table with indexes from combination tables
results = pd.DataFrame(index=combinations.index, columns=('training_time', 'test_time', 'n_train', 'n_test', 'score', 'feat_vect_length','error', 'error_msg'))

for i in tqdm.tqdm(range(len(combinations))):

    c = combinations.iloc[i]
    try:
        features = extract_features(images, color_space=c['color_space'],
                                    spatial_size=c['spatial_size'], hist_bins=c['hist_bins'],
                                    orient=c['orient'], pix_per_cell=c['pix_per_cell'],
                                    cell_per_block=c['cell_per_block'],
                                    hog_channel=c['hog_channel'],
                                    spatial_feat=c['spatial_feat'],
                                    hist_feat=c['hist_feat'], hog_feat=c['hog_feat'])

        X = np.array(features).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)


        # Split up data into randomized training and test sets
        rand_state = 888
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.3, random_state=rand_state)
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=888)

        # Use a linear SVC
        svc = LinearSVC()

        # Check the training time for the SVC
        start_training = time.time()
        svc.fit(X_train, y_train)
        end_training = time.time()

        # Check the score of the SVC (Only used as timing purposes)
        fit_score = svc.score(X_test, y_test)

        # Check the prediction time for a single sample
        end_prediction = time.time()

        # Calculate cross valitated score (this is more reliable way)
        score = cross_val_score(svc, scaled_X, y, cv=sss,n_jobs=2).mean()

        # ('training_time', 'test_time', 'n_train', 'n_test' 'score', 'error', 'error_msg'))
        results.iloc[i] = [(end_training-start_training), (end_prediction-end_training), len(X_train), len(X_test), score, len(features[1]), False, ""]

    except Exception as e:
        results.iloc[i] = [None, None, None, None, None, None, True, e]

    # Save results every Nth iteration
    if ((i % 50) == 0) & save_results:
        all = results.join(combinations)
        all.to_csv(filename)

# Save final results
if save_results:
    all = results.join(combinations)
    all.to_csv(filename)
