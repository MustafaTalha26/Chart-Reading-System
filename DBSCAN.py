from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscanT(X,n):
    X = StandardScaler().fit_transform(X)
    clustering = DBSCAN(eps=0.5, min_samples=n).fit(X)
    return clustering.labels_