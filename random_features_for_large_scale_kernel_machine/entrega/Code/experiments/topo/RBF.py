import numpy as np
# Data sets
from sklearn.datasets import fetch_covtype
import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import linear_model
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.datasets import fetch_openml

print('Random Binning Features.+ LS with  TOPO data set')

topo = fetch_openml(name='topo_2_1', version=1)
test_size_ratio = 0.3
print('Size: ', topo.data.shape)
print('Target size: ', topo.target.shape)

X_train, X_test, y_train, y_test = train_test_split(
    topo.data, topo.target, test_size= test_size_ratio, random_state=42)

# Adjust 
for D in [100]: #, 200, 500]:
    for g in [0.05, 0.5, 1]:
        est = KBinsDiscretizer(n_bins=D, encode='ordinal',
                                strategy='uniform')
        # Get best alpha
        X_features = est.fit_transform(X_train)
        clf = linear_model.RidgeCV(alphas=np.linspace(0.05, 10, 10))
        clf.fit(X_features, y_train)
        best_alpha = clf.alpha_
        
        # Adjust the model with the best hyperparameters and control time
        clf = linear_model.Ridge(alpha=best_alpha)
        init_time = timeit.default_timer()
        X_features = est.fit_transform(X_train)
        clf.fit(X_features, y_train)
        end_time = timeit.default_timer()
        spent_time = end_time - init_time

        print(f'Time spent in RBF + LS D={D} gamma = {g}  alpha = {best_alpha:.3f}: {spent_time:.4f} s')
        X_features_test = est.transform(X_test)

        score = clf.score(X_features_test, y_test)
        print(f'Score RBF + LS D={D} gamma = {g} R: {score :.4f}\n')