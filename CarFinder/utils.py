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
        x.append(cv2.imread(img_file))
        y.append(1)  # Could be also replaced by np.ones() outside of the loop

    for img_file in nonvehicles:
        x.append(cv2.imread(img_file))
        y.append(
            0)  # Could be also replaced by np.zeros() outside of the loop

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