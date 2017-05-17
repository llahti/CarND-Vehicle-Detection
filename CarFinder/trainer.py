
if __name__ == "__main__":
    from CarFinder.classifier import Classifier
    from CarFinder.features import Features
    from CarFinder.utils import get_module_path, load_images
    import glob
    import dataset

    basepath = get_module_path()
    print(basepath)

    vehicles = glob.glob('../data/vehicles/**/*.png')
    nonvehicles = glob.glob('../data/non-vehicles/**/*.png')
    print("Training set totally contains {} vehicle images and {} non-vehicle images".format(len(vehicles), len(nonvehicles)))
    images, y = load_images(vehicles, nonvehicles)
    print("After loading data set consists of {} images and {} labels".format(len(images), len(y)))

    print("Extract Features")
    f = Features()
    X = f.extract_features(images)
    print("Feature vector length: {}".format(X.shape[1]))

    #Create classifier
    print("Create classifier object")
    c = Classifier(True, True)
    print("Fit Classifier")
    c.fit(X, y, True)

    # Save default scaler and classifier
    print("Save default scaler and classifier")
    c.save_scaler(basepath + "/default_scaler.pkl")
    c.save_classifier(basepath + "/default_classifier.pkl")


