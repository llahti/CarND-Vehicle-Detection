import glob
import cv2
import numpy as np

def load_small():
    """
    Returns features and labels of small training set.
    :return: (x, y) Images are in format uint8 BGR
    """
    vehicles = glob.glob('./data/vehicles_smallset/**/*.jpeg', recursive=True)
    nonvehicles = glob.glob('./data/non-vehicles_smallset/**/*.jpeg', recursive=True)

    x, y = [], []
    for img_file in vehicles:
        x.append(cv2.imread(img_file))
        y.append(1)  # Could be also replaced by np.ones() outside of the loop

    for img_file in nonvehicles:
        x.append(cv2.imread(img_file))
        y.append(0)  # Could be also replaced by np.zeros() outside of the loop

    # Convert to numpy arrays and return
    return np.array(x), np.array(y)


def load_full():
    """
    Returns features and labels of full training set.
    :return: (x, y) Images are in format uint8 BGR
    """
    vehicles = glob.glob('./data/vehicles/**/*.png', recursive=True)
    nonvehicles = glob.glob('./data/non-vehicles/**/*.png', recursive=True)

    x, y = [], []
    for img_file in vehicles:
        x.append(cv2.imread(img_file))
        y.append(1)  # Could be also replaced by np.ones() outside of the loop

    for img_file in nonvehicles:
        x.append(cv2.imread(img_file))
        y.append(0)  # Could be also replaced by np.zeros() outside of the loop

    # Convert to numpy arrays and return
    return np.array(x), np.array(y)


def print_statistics():
    """
    This function prints statistics of available data sets
    :return: 
    """

    # Load the subset of training data
    x_small, y_small = load_small()
    print("Small data set:")
    # Get number of vehicle training images
    print("\tNumber of vehicle images is {}".format(len(y_small[y_small == 1])))
    # get number of non vehicle training images
    print("\tNumber of non-vehicle images is {}".format(len(y_small[y_small == 0])))
    # Shape
    print("\tShape of the image is ", x_small[0].shape)
    # Datatype
    print("\tDatatype of image is ", x_small[0].dtype)

    # Load the full training data set
    x_full, y_full = load_full()
    print("\n\nFull data set:")
    # Get number of vehicle training images
    print("\tNumber of vehicle images is {}".format(len(y_full[y_full == 1])))
    # get number of non vehicle training images
    print("\tNumber of non-vehicle images is {}".format(len(y_full[y_full == 0])))
    # Shape
    print("\tShape of the image is ", x_full[0].shape)
    # Datatype
    print("\tDatatype of image is ", x_full[0].dtype)

if __name__ == '__main__':
    print_statistics()
    x, y = load_small()
    print(x.shape)