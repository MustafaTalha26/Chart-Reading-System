from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kmeansT(X,n):
    X = StandardScaler().fit_transform(X)
    clustering = KMeans(n_clusters=n,n_init=8).fit(X)
    return clustering.labels_