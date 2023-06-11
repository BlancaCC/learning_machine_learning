import numpy as np
# Data sets
from sklearn.datasets import fetch_covtype
import timeit
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_openml

print('Random Fourier Features + LS with  TOPO data set')


topo = fetch_openml(name='topo_2_1', version=1)
test_size_ratio = 0.3
print('Size: ', topo.data.shape)
print('Target size: ', topo.target.shape)

X_train, X_test, y_train, y_test = train_test_split(
    topo.data, topo.target, test_size= test_size_ratio, random_state=42)

scaler = preprocessing.StandardScaler().fit(X_train)

# Adjust 


for D in [100, 200, 500, 1000]:
    for g in [0.05, 0.5, 1]:
        rbf_feature = RBFSampler(gamma=g, n_components=D, random_state=1)
        # Get best alpha
        clf = linear_model.RidgeCV(alphas=np.linspace(0.05, 10, 10))
        X_features = rbf_feature.fit_transform(X_train)
        clf.fit(X_features, y_train)
        best_alpha = clf.alpha_
        
        # Adjust the model with the best hyperparameters and control time
        clf = linear_model.Ridge(alpha=best_alpha)
        init_time = timeit.default_timer()
        X_features = rbf_feature.fit_transform(X_train)
        clf.fit(X_features, y_train)
        end_time = timeit.default_timer()
        spent_time = end_time - init_time

        print(f'Time spent in RFF D={D} gamma = {g}  alpha = {best_alpha:.3f}: {spent_time:.4f} s')
        
        X_features_test = rbf_feature.transform(X_test)
        score = clf.score(X_features_test, y_test)
        print(f'Score RFF D={D} gamma = {g} R: {score :.4f}\n')