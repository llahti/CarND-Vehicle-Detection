import glob
import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import CarFinder.utils as utils
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


class Classifier:
    def __init__(self, classifier=None, scaler=None):
        """
        Initializes classifier.
        
        :param classifier: sklearn classifier. If not given then load the default classifier. 
        :param scaler:  sklearn scaler. If not given then load default scaler.
        """

        basepath = utils.get_module_path()
        if not scaler:
            self.load_scaler(basepath + "/default_scaler.pkl")
        if not classifier:
            self.load_classifier(basepath + "/default_classifier.pkl")

    def load_scaler(self, path):
        """
        Load feature scaler from file.
        
        :param path: 
        :return: 
        """
        self.scaler = joblib.load(path)

    def save_scaler(self, path):
        """
        Save feature scaler into file.
        
        :param path: 
        :return: 
        """
        joblib.dump(self.scaler, path)

    def load_classifier(self, path):
        """
        Loads classifier from file.
                
        :param path: 'filename.pkl'
        :return: 
        """
        self.classifier = joblib.load(path)

    def save_classifier(self, path):
        """
        Save classifier into file
        
        :param path: 'filename.pkl' 
        :return: 
        """
        joblib.dump(self.classifier, path)

    def predict(self, features):
        return self.classifier.predict(features)

    def fit(self, X, y, verbose=False):
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)

        # Scale X
        X = self.transform(X)

        # Set statified crossvalidation for grid search
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,
                                     random_state=888)

        # kernel: 'rbf', C: 10, gamma: 0.0002
        #self.classifier = SVC(kernel='rbf', C=10, gamma=0.0002, verbose=verbose)
        self.classifier = SVC(kernel='linear', C=50, verbose=verbose)


        # Calculate cross valitated score (this is more reliable way)
        score = cross_val_score(self.classifier, X, y, cv=sss, n_jobs=1).mean()

        # Train with all available data
        self.classifier.fit(X, y)

        return score

    def transform(self, features_vect):
        """
        Scale data.
        
        :param features_vect: 
        :return: 
        """
        scaled_X = self.scaler.transform(features_vect)
        return scaled_X


if __name__ == "__main__":
    from CarFinder.features import Features
    import CarFinder.utils as utils
    import tempfile
    from sklearn.model_selection import StratifiedShuffleSplit

    c = Classifier()
    f = Features()
    images, labels = utils.load_test_images()
    features = f.extract_features(images)

    # Test training
    scores_a = c.fit(features, labels)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3,
                                 random_state=888)
    score_a = cross_val_score(c.classifier, features, labels, cv=sss, n_jobs=1).mean()

    print("Cross_val_score=", score_a)

    tmp_clf = tempfile.TemporaryFile()
    tmp_scl = tempfile.TemporaryFile()
    # Save and load classifier and scaler
    c.save_classifier(tmp_clf)
    c.save_scaler(tmp_scl)
    c.load_classifier(tmp_clf)
    c.load_scaler(tmp_scl)

    score_b = cross_val_score(c.classifier, features, labels, cv=sss, n_jobs=1).mean()
    print("Cross_val_score=", score_b)

