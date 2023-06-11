# utils 
from sklearn.model_selection import train_test_split
import timeit
# models
from sklearn import svm
# Data sets

from sklearn.datasets import fetch_openml
topo = fetch_openml(name='topo_2_1', version=1)

print('SVM for topo_2_1 dataset (for regression)')

test_size_ratio = 0.3
print('Size: ', topo.data.shape)
print('Target size: ', topo.target.shape)

X_train, X_test, y_train, y_test = train_test_split(
    topo.data, topo.target, test_size= test_size_ratio, random_state=42)

# Training 
clf = svm.SVR(gamma='scale') # regression problem
init_time = timeit.default_timer()
clf.fit(X_train, y_train)
end_time = timeit.default_timer()
spent_time = end_time - init_time
print(f'Time spent in SVR regression{spent_time:.4f} s')
# Test 
score = clf.score(X_test, y_test)
print(f'Score R^2{score :.4f}')