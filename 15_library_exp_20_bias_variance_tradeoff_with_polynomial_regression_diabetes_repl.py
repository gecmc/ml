"""Exp 20. Bias-variance tradeoff with polynomial regression (Diabetes replaces Boston)"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_diabetes(return_X_y=True)
X=X[:,[2,3,8]]
Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=0.3,random_state=0)

d=range(1,7)
tr_e,va_e=[],[]
for k in d:
    m=make_pipeline(StandardScaler(),PolynomialFeatures(k),LinearRegression()).fit(Xtr,ytr)
    tr_e.append(mean_squared_error(ytr,m.predict(Xtr)))
    va_e.append(mean_squared_error(yva,m.predict(Xva)))
    print(f"Degree {k} | Train MSE: {tr_e[-1]:.2f} | Val MSE: {va_e[-1]:.2f}")

plt.plot(d,tr_e,"o-",label="Train");plt.plot(d,va_e,"o-",label="Validation")
plt.xlabel("Degree");plt.ylabel("MSE (log scale)");plt.yscale("log");plt.legend();plt.show()
