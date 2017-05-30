import cv2
import os
import numpy as np
import glob


def load_images(vehicles, nonvehicles):
    """
    Load vehicle and non vehicle images
    :param vehicles: List on vehicle images on disk
    :param nonvehicles: list of non-vehicle images on disk
    :return: array of images (vehicles, non-vehicles)
    """
    x, y = [], []
    for img_file in vehicles:
        img = cv2.imread(img_file)
        if img is None:
            raise Exception("Image {} is bad".format(img_file))
        x.append(img)
        y.append(1)  # Could be also replaced by np.ones() outside of the loop

    for img_file in nonvehicles:
        img = cv2.imread(img_file)
        if img is None:
            raise Exception("Image {} is bad".format(img_file))
        x.append(img)
        y.append(0)  # Could be also replaced by np.zeros() outside of the loop

    # Convert to numpy arrays and return
    return np.array(x), np.array(y)


def get_module_path():
    """
    This function returns absolute path of module. In other words the 
    location of __init__.py file.
    """
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    return dir_path


def load_test_images():
    """
    This function loads test data set containing 10 vehicle images and 
    10 non-vehicle images
    
    :return: images, labels 
    """
    base_path = get_module_path()
    vehicles_pattern = base_path + '/test_images/vehicles/*.png'
    nonvehicles_pattern = base_path + '/test_images/nonvehicles/*.png'


    # Load image paths
    print("Reading test vehicle test images from path:", vehicles_pattern)
    vehicles = glob.glob(vehicles_pattern,
                         recursive=True)
    print("Read {} images".format(len(vehicles)))
    print("Reading non-vehicle test images from path:", nonvehicles_pattern)
    nonvehicles = glob.glob(nonvehicles_pattern,
                            recursive=True)
    print("Read {} images".format(len(nonvehicles)))

    images, labels = load_images(vehicles, nonvehicles)
    print("Totally loaded {} images to training set".format(len(labels)))

    return images, labels


def pip(image, subimage, pos, size, border=5, title=""):
    """
    Adds sub image into image on given position and given size.

    :param image: Image to where subimage is placed 
    :param subimage: Image to be placed into image
    :param pos: Position of top left corner of subimage in image (x, y)
    :param size: Size of the subimage (It'll be resized to this size) (x, y)
    :param border: thickness of black border around subimage
    :param title: Title of pip window (shown on upper left corner of pip window)
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

    if len(title) != 0:
        cv2.putText(image, title, (x_left, y_top + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    return image