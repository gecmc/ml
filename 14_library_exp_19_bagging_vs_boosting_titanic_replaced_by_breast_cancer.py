"""Exp 19. Bagging vs boosting (Titanic replaced by Breast Cancer)"""

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)

models={"Bagging":BaggingClassifier(DecisionTreeClassifier(),n_estimators=50,random_state=0),
        "Boosting":AdaBoostClassifier(n_estimators=50,random_state=0)}
for n,m in models.items():
    p=m.fit(Xtr,ytr).predict(Xte)
    print(f"{n:<9} | Acc: {accuracy_score(yte,p):.4f} | Prec: {precision_score(yte,p):.4f} | Rec: {recall_score(yte,p):.4f} | F1: {f1_score(yte,p):.4f}")
