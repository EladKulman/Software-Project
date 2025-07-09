import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def plot_elbow_method():
    iris = load_iris()
    X = iris.data

    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.xlabel('K')
    plt.ylabel('Average Dispersion (Inertia)')
    plt.xticks(range(1, 11))
    plt.grid(True)

    # Annotate the elbow point (example, typically around K=3)
    # This is a visual estimation, for this dataset, 3 is a common elbow point.
    plt.annotate('Elbow Point', xy=(3, inertias[2]), xytext=(4, inertias[2] + 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig('elbow.png')
    plt.close()

if __name__ == "__main__":
    plot_elbow_method()
