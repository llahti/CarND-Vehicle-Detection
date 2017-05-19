from CarFinder.features import Features
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import cv2
import dataset

# Generate spatial binning illustration
def show_spatial_binning():
    X, y = dataset.load_small()
    f = Features()
    img = X[0]
    binned = cv2.resize(img, f.spatial_binning_size)
    binned_ravel = binned.copy().ravel()

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Spatial Binning", fontsize=20)

    a = fig.add_subplot(1, 3, 1)
    plt.title("Original (64, 64, 3)")
    plt.imshow(img, cmap='gray')

    a = fig.add_subplot(1, 3, 2)
    plt.title("Resized (15, 15, 3)")
    plt.imshow(binned, cmap='gray')

    a = fig.add_subplot(1, 3, 3)
    plt.title("1D Feature Vector")
    plt.plot(binned_ravel)
    plt.show()


def show_color_histogram():
    X, y = dataset.load_small()
    f = Features()
    img = X[10]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    hist_ch1 = np.histogram(img[:, :, 0], bins=f.hist_nbins,
                            range=(0, 255))
    hist_ch2 = np.histogram(img[:, :, 1], bins=f.hist_nbins,
                            range=(0, 255))
    hist_ch3 = np.histogram(img[:, :, 2], bins=f.hist_nbins,
                            range=(0, 255))

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Color Histogram", fontsize=20)

    a = fig.add_subplot(1, 4, 1)
    plt.title("Channel 1")
    # plt.imshow(img, cmap='gray')
    plt.plot(hist_ch1[0])

    a = fig.add_subplot(1, 4, 2)
    plt.title("Channel 2")
    # plt.imshow(binned, cmap='gray')
    plt.plot(hist_ch2[0])

    a = fig.add_subplot(1, 4, 3)
    plt.title("Channel 3")
    plt.plot(hist_ch3[0])

    a = fig.add_subplot(1, 4, 4)
    plt.title("1D Feature Vector")
    plt.plot(np.concatenate((hist_ch1[0], hist_ch2[0], hist_ch3[0])))
    plt.show()

def show_hog_features():
    X, y = dataset.load_small()
    f = Features()
    img = X[10]

    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    hog_vect_ch1, hog_img_ch1 = f.get_hog_features(img[:,:,0], vis=True, feature_vec=True)
    hog_vect_ch2, hog_img_ch2 = f.get_hog_features(img[:, :, 1], vis=True,
                                                   feature_vec=True)
    hog_vect_ch3, hog_img_ch3 = f.get_hog_features(img[:, :, 2], vis=True,
                                                   feature_vec=True)


    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Histogram of Oriented Gradients", fontsize=20)

    a = fig.add_subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(orig, cmap='gray')
    #plt.plot(hist_ch1[0])

    a = fig.add_subplot(2, 4, 2)
    plt.title("HOG - Channel 1")
    plt.imshow(hog_img_ch1, cmap='gray')

    a = fig.add_subplot(2, 4, 3)
    plt.title("HOG - Channel 2")
    plt.imshow(hog_img_ch2, cmap='gray')

    a = fig.add_subplot(2, 4, 4)
    plt.title("HOG - Channel 3")
    plt.imshow(hog_img_ch3, cmap='gray')

    a = fig.add_subplot(2, 4, 5)
    plt.title("1D Feature Vector")
    #plt.imshow(hog_img_ch3, cmap='gray')
    plt.plot(np.concatenate((hog_vect_ch1, hog_vect_ch2, hog_vect_ch3)).ravel())

    plt.show()


if __name__ == "__main__":
    #show_spatial_binning()
    #show_color_histogram()
    show_hog_features()