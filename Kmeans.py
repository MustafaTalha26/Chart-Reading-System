from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kmeansT(X,n):
    X = StandardScaler().fit_transform(X)
    clustering = KMeans(n_clusters=n).fit(X)
    return clustering.labels_