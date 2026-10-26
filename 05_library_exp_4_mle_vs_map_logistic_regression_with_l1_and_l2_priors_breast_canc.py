"""Exp 4. MLE vs MAP logistic regression with L1 and L2 priors (Breast Cancer)"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)

models={"MLE":LogisticRegression(penalty=None,max_iter=5000),
        "MAP-L2 (Gaussian prior)":LogisticRegression(penalty="l2",C=0.1,max_iter=5000),
        "MAP-L1 (Laplace prior)":LogisticRegression(penalty="l1",C=0.1,solver="liblinear")}
for n,m in models.items():
    m.fit(Xtr,ytr)
    print(f"{n:<25} | Acc: {accuracy_score(yte,m.predict(Xte)):.4f} | Sum|w|: {abs(m.coef_).sum():.2f} | Zero weights: {(m.coef_==0).sum()}")
