"""Exp 12. SVM with linear, polynomial and RBF kernels"""

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)

for k in ["linear","poly","rbf"]:
    m=SVC(kernel=k).fit(Xtr,ytr)
    print(f"{k:<7} | Accuracy: {accuracy_score(yte,m.predict(Xte)):.4f}")
