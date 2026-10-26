"""Exp 17. K-means on Digits for different K (inertia and silhouette)"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X,_=load_digits(return_X_y=True)
ks=range(2,16)
inertia,sil=[],[]
for k in ks:
    m=KMeans(k,n_init=10,random_state=0).fit(X)
    inertia.append(m.inertia_)
    sil.append(silhouette_score(X,m.labels_))
    print(f"K={k:<3} | Inertia: {m.inertia_:.2f} | Silhouette: {sil[-1]:.4f}")

fig,ax=plt.subplots(1,2)
ax[0].plot(ks,inertia,"o-");ax[0].set_title("Inertia")
ax[1].plot(ks,sil,"o-");ax[1].set_title("Silhouette")
plt.show()
