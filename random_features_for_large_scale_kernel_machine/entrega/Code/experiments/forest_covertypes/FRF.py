import numpy as np
# Data sets
from sklearn.datasets import fetch_covtype
import timeit
from config import test_size_ratio
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn import linear_model

# Get data
X,y = fetch_covtype(return_X_y=True)
print('Size: ', X.shape)
print('Target size: ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= test_size_ratio, random_state=42) 

# Adjust 
print('Random Fourier Features with Forest covertypes data set')
for D in [20, 50, 100, 200, 500]:
    for g in [0.05, 0.5, 1]:
        rbf_feature = RBFSampler(gamma=g, n_components=D, random_state=1)
        # Get best alpha
        clf = linear_model.RidgeClassifierCV()
        X_features = rbf_feature.fit_transform(X_train)
        clf.fit(X_features, y_train)
        best_alpha = clf.alpha_
        
        # Adjust the model with the best hyperparameters and control time
        clf = linear_model.RidgeClassifier(alpha=best_alpha)
        init_time = timeit.default_timer()
        X_features = rbf_feature.fit_transform(X_train)
        clf.fit(X_features, y_train)
        end_time = timeit.default_timer()
        spent_time = end_time - init_time

        print(f'Time spent in RFF D={D} gamma = {g} : {spent_time:.4f} s')
        X_features_test = rbf_feature.fit_transform(X_test)

        score = clf.score(X_features_test, y_test)
        print(f'Score RFF D={D} gamma = {g}: {score :.4f}')