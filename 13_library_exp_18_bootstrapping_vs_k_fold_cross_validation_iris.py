"""Exp 18. Bootstrapping vs k-fold cross-validation (Iris)"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X,y=load_iris(return_X_y=True)
m=DecisionTreeClassifier(random_state=0)
ids=np.arange(len(y))

acc=[]
for i in range(50):
    tr=resample(ids,random_state=i)
    oob=np.setdiff1d(ids,tr)
    acc.append(m.fit(X[tr],y[tr]).score(X[oob],y[oob]))
print(f"Bootstrap  | Accuracy: {np.mean(acc):.4f} +/- {np.std(acc):.4f}")

cv=cross_val_score(m,X,y,cv=5)
print(f"5-fold CV  | Accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
