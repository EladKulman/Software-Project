import sys
import numpy as np
import pandas as pd
import mykmeanssp

def validate_input(k, iter, eps, file1, file2):
    if not (isinstance(k, int) and k > 1):
        print("Invalid number of clusters!")
        return False
    if not (isinstance(iter, int) and 1 < iter < 1000):
        print("Invalid maximum iteration!")
        return False
    if not (isinstance(eps, float) and eps >= 0):
        print("Invalid epsilon!")
        return False
    return True

def kmeans_pp(k, iter, eps, file1, file2):
    np.random.seed(1234)

    try:
        df1 = pd.read_csv(file1, header=None)
        df2 = pd.read_csv(file2, header=None)
    except FileNotFoundError:
        print("An Error Has Occurred")
        sys.exit(1)

    # Inner join and sort
    merged_df = pd.merge(df1, df2, on=0, how='inner')
    merged_df = merged_df.sort_values(by=0)

    indices = merged_df.iloc[:, 0].astype(int).tolist()
    datapoints = merged_df.iloc[:, 1:].to_numpy(dtype=np.float64)

    N, d = datapoints.shape

    if k >= N:
        print("Invalid number of clusters!")
        sys.exit(1)

    # K-means++ initialization
    centroid_indices = np.zeros(k, dtype=int)
    centroids = np.zeros((k, d))

    # Step 1: Choose first centroid uniformly at random
    initial_index = np.random.choice(N)
    centroid_indices[0] = indices[initial_index]
    centroids[0] = datapoints[initial_index]

    D = np.full(N, np.inf)

    for i in range(1, k):
        # Step 2: Compute distances to the nearest centroid
        for j in range(N):
            dist = np.linalg.norm(datapoints[j] - centroids[i-1])
            D[j] = min(D[j], dist)
        
        # Step 3: Choose new centroid with weighted probability
        D_squared = D**2
        probabilities = D_squared / np.sum(D_squared)
        new_centroid_idx_in_datapoints = np.random.choice(N, p=probabilities)
        
        centroid_indices[i] = indices[new_centroid_idx_in_datapoints]
        centroids[i] = datapoints[new_centroid_idx_in_datapoints]

    # Call C extension
    initial_centroids_list = centroids.tolist()
    datapoints_list = datapoints.tolist()

    final_centroids = mykmeanssp.fit(initial_centroids_list, datapoints_list, k, iter, eps, N, d)

    # Output results
    print(",".join(map(str, centroid_indices)))
    for centroid in final_centroids:
        print(",".join([f"{x:.4f}" for x in centroid]))


def main():
    # Argument parsing
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        if len(sys.argv) == 6:
            iter = int(sys.argv[2])
            eps = float(sys.argv[3])
            file1 = sys.argv[4]
            file2 = sys.argv[5]
        else:
            iter = 300
            eps = float(sys.argv[2])
            file1 = sys.argv[3]
            file2 = sys.argv[4]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    if not validate_input(k, iter, eps, file1, file2):
        sys.exit(1)

    kmeans_pp(k, iter, eps, file1, file2)

if __name__ == "__main__":
    main()
