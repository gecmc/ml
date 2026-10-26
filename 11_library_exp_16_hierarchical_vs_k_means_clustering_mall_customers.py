"""Exp 16. Hierarchical vs K-means clustering (Mall Customers)"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score

X,_=make_blobs(n_samples=200,centers=5,random_state=0)
km=KMeans(5,n_init=10,random_state=0).fit(X)
hc=AgglomerativeClustering(5).fit(X)

print(f"KMeans       | Inertia: {km.inertia_:.2f} | Silhouette: {silhouette_score(X,km.labels_):.4f}")
print(f"Hierarchical | Silhouette: {silhouette_score(X,hc.labels_):.4f}")
fig,ax=plt.subplots(1,2)
ax[0].scatter(X[:,0],X[:,1],c=km.labels_);ax[0].set_title("KMeans")
ax[1].scatter(X[:,0],X[:,1],c=hc.labels_);ax[1].set_title("Hierarchical")
plt.show()
