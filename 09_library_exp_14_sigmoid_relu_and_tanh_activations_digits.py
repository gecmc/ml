"""Exp 14. Sigmoid, ReLU and Tanh activations (Digits)"""

import time
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)

for a in ["logistic","relu","tanh"]:
    t=time.time()
    m=MLPClassifier((64,),activation=a,max_iter=500,random_state=0).fit(Xtr,ytr)
    print(f"{a:<9} | Acc: {accuracy_score(yte,m.predict(Xte)):.4f} | Epochs: {m.n_iter_} | Time: {time.time()-t:.2f}s")
