from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

treescores_in = []
treescores_out = []

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = i, stratify = y)
#    sc = StandardScaler()
#    sc.fit(X_train)
#    X_train_std = sc.transform(X_train)
#    X_test_std = sc.transform(X_test)

    tree = DecisionTreeClassifier(criterion = 'gini')
    tree.fit(X_train, y_train)
    y_pred_in = tree.predict(X_train)
    y_pred_out = tree.predict(X_test)
    treescores_in.append(accuracy_score(y_train, y_pred_in))
    treescores_out.append(accuracy_score(y_test, y_pred_out))
    print('random_state = ', i, ', accuracy score of treeclassifier     in sample= ', treescores_in[i-1])
    print('random_state = ', i, ', accuracy score of treeclassifier out of sample= ', treescores_out[i-1])

print('\nin-sample accuracy score: %.3f +/- %.3f'%(np.mean(treescores_in),np.std(treescores_in)))
print('\nout-of-sample accuracy score: %.3f +/- %.3f'%(np.mean(treescores_out),np.std(treescores_out)))

#k_fold_accu = []
#from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import make_pipeline
#from sklearn import metrics
#
#for i in range(1,11):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = i, stratify = y)
#    pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
#    pipe_lr.fit(X_train, y_train)
#    y_pred = pipe_lr.predict(X_test)
#    k_fold_accu.append(metrics.accuracy_score(y_test, y_pred))
#print(k_fold_accu)
#
#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)
#print(scores)
#print('\nCV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



print("My name is {Yue Liu}")
print("My NetID is: {yueliu6}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")