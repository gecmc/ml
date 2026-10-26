"""Q3. Logistic regression with L2 regularization, with and without scaling (Breast Cancer)"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sig=lambda z:1/(1+np.exp(-np.clip(z,-500,500)))

def fit(X,y,lr=0.1,lam=0.1,n=1000):
    w,b=np.zeros(X.shape[1]),0.0
    for _ in range(n):
        p=sig(X@w+b)
        w-=lr*(X.T@(p-y)/len(y)+lam*w)
        b-=lr*(p-y).mean()
    return w,b

def run(A,B,tag,lr):
    w,b=fit(A,ytr,lr)
    p=(sig(B@w+b)>0.5).astype(int)
    print(f"{tag:<9} | Acc: {accuracy_score(yte,p):.4f} | Prec: {precision_score(yte,p):.4f} | Rec: {recall_score(yte,p):.4f} | F1: {f1_score(yte,p):.4f}")

run(Xtr,Xte,"Unscaled",0.001)
sc=StandardScaler().fit(Xtr)
run(sc.transform(Xtr),sc.transform(Xte),"Scaled",0.1)
