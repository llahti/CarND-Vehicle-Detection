"""This script is used to train classifier and save classifier and scaler onto 
disk as a defauls for Classifier-class."""

if __name__ == "__main__":
    from CarFinder.classifier import Classifier
    from CarFinder.features import Features
    from CarFinder.utils import get_module_path, load_images
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    import glob
    import dataset

    basepath = get_module_path()
    print(basepath)

    vehicles = glob.glob('../data/vehicles/**/*.png')
    nonvehicles = glob.glob('../data/non-vehicles/**/*.png')
    #vehicles = glob.glob('../data/vehicles_smallset/**/*.jpeg')
    #nonvehicles = glob.glob('../data/non-vehicles_smallset/**/*.jpeg')
    #vehicles = glob.glob('../data/vehicles_selected/**/*.png')
    #nonvehicles = glob.glob('../data/non-vehicles_selected/**/*.png')
    print("Training set totally contains {} vehicle images and {} non-vehicle images".format(len(vehicles), len(nonvehicles)))
    images, y = load_images(vehicles, nonvehicles)
    print("After loading data set consists of {} images and {} labels".format(len(images), len(y)))

    print("Extract Features")
    f = Features()
    X = f.extract_features(images)
    print("feature set min {} and max {}".format(X.min(), X.max()))
    print("Feature vector length: {}".format(X.shape[1]))

    #Create classifier
    print("Create classifier object")
    c = Classifier(True, True)
    print("Fit Classifier")
    #samples = 2000
    #X, y = shuffle(X, y, random_state=123)
    #c.fit(X[:samples], y[:samples], True)
    c.fit(X, y, True)

    # Save default scaler and classifier
    print("\nSave default scaler and classifier")
    c.save_scaler(basepath + "/default_scaler.pkl")
    c.save_classifier(basepath + "/default_classifier.pkl")

    print("Calculate cross-validation scores")
    # Calculate cross validation score
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=888)
    # Calculate cross validated score (this is more reliable way)
    scaled_X = c.transform(X)
    score = cross_val_score(c.classifier, scaled_X, y, cv=sss, n_jobs=8).mean()
    print("Cross validation score is {}".format(score))
