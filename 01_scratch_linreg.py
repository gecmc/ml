"""Q1. Univariate linear regression with gradient descent (Diabetes, BMI) and learning rate decay"""

import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

X,y=load_diabetes(return_X_y=True)
X=X[:,[2]]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr)[:,0],sc.transform(Xte)[:,0]

w,b,lr,n=0.0,0.0,0.1,200
costs=[]
for e in range(n):
    a=lr/(1+0.01*e)
    err=w*Xtr+b-ytr
    w-=a*(err@Xtr)/len(ytr)
    b-=a*err.mean()
    costs.append((err**2).mean()/2)

print(f"w = {w:.4f} | b = {b:.4f}")
print(f"MSE = {mean_squared_error(yte,w*Xte+b):.2f}")
plt.plot(costs);plt.xlabel("Epoch");plt.ylabel("Cost");plt.show()
