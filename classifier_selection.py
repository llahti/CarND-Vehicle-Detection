"""
http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

"""


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from dataset import *
import pandas as pd
from CarFinder.features import Features


save_results = True
#filename = './model_selection/classifier_results.csv'
filename = './model_selection/classifier_results_temp.csv'

print("Loading data... ", end='')
images, y = load_full()
print("OK!")

print("Extracting Features... ", end='')
f = Features()  # Use defaults
features = f.extract_features(images)
print("OK!")


print("Scaling features...", end='')
# Ensure that datatype is float64 (it is needed by scaler)
X = np.array(features).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)
print("OK!")

# Set statified crossvalidation for grid search
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=888)

param_grid = [
  #{'C': [100,], 'gamma': [0.0006, 0.0007, 0.0008, 0.0009, 0.001 ], 'kernel': ['rbf']},
  {'C': [1, 10, 100], 'gamma': np.linspace(0.0001, 0.0005, 5), 'kernel': ['rbf']},
  {'C': [1, 10, 20, 50], 'kernel': ['linear',]}
 ]

svr = SVC()
print("Doing grid search...", end='')
clf = GridSearchCV(svr, param_grid, cv=sss, n_jobs=8,verbose=2)
clf.fit(scaled_X, y)
print("OK!")

print("Best parameters are")
print(clf.best_params_)
print("\n\n")

print(clf.cv_results_)
print()


print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
#fit_time = clf.cv_results_['fit_time']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.5f (+/-%0.03f) t=%0.5f for %r"
          % (mean, std * 2, 0, params))
print()

df = pd.DataFrame(clf.cv_results_)
df.to_csv(filename)