"""Exp 3. Ridge and Lasso vs linear regression, tuned with cross-validation (Diabetes)"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_diabetes(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)

models={"Linear":LinearRegression(),
        "Ridge":RidgeCV(alphas=[0.01,0.1,1,10],cv=5),
        "Lasso":LassoCV(cv=5)}
for n,m in models.items():
    p=m.fit(Xtr,ytr).predict(Xte)
    print(f"{n:<7} | MSE: {mean_squared_error(yte,p):.2f} | R2: {r2_score(yte,p):.4f}")
