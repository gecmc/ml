"""Exp 15. Hyperparameter tuning: learning rate, batch size, epochs (Digits)"""

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_digits(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)

for lr in [0.001,0.01]:
    for bs in [32,128]:
        for ep in [20,100]:
            m=MLPClassifier((64,),learning_rate_init=lr,batch_size=bs,max_iter=ep,random_state=0).fit(Xtr,ytr)
            print(f"lr={lr:<6} batch={bs:<4} epochs={ep:<4} | Accuracy: {accuracy_score(yte,m.predict(Xte)):.4f}")
