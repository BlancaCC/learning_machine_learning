# utils 
from sklearn.model_selection import train_test_split
# models
from sklearn import svm
# Data sets
from sklearn.datasets import fetch_covtype
import timeit
from config import test_size_ratio
# Get data
X,y = fetch_covtype(return_X_y=True)
print('Size: ', X.shape)
print('Target size: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= test_size_ratio, random_state=42) 

# Training 
clf = svm.SVC(decision_function_shape='ovo') #multiclass
init_time = timeit.default_timer()
clf.fit(X_train, y_train)
end_time = timeit.default_timer()
spent_time = end_time - init_time
print(f'Time spent in SVM multiclass {spent_time:.4f} s')
# Test 
score = clf.score(X_test, y_test)
print(f'Score {score :.4f}')