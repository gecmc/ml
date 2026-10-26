"""Q2. KNN from scratch on Digits with Euclidean and Manhattan distance"""

import numpy as np, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)

euc=lambda a,b:np.sqrt(((a-b)**2).sum(1))
man=lambda a,b:np.abs(a-b).sum(1)

def knn(x,k,dist):
    idx=np.argsort(dist(Xtr,x))[:k]
    return np.bincount(ytr[idx]).argmax()

for name,d in [("Euclidean",euc),("Manhattan",man)]:
    for k in [1,3,5,7,9]:
        t=time.time()
        p=[knn(x,k,d) for x in Xte]
        print(f"{name:<10} K={k} | Accuracy: {accuracy_score(yte,p):.4f} | Time: {time.time()-t:.2f}s")
