import sys
import numpy as np
import pandas as pd

def parse_arguments():
    if len(sys.argv) not in (5, 6):
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(float(sys.argv[1]))
    except ValueError:
        print("Invalid number of clusters!")
        sys.exit(1)

    if len(sys.argv) == 6:
        try:
            max_iter = int(float(sys.argv[2]))
        except ValueError:
            print("Invalid maximum iteration!")
            sys.exit(1)
        try:
            eps = float(sys.argv[3])
        except ValueError:
            print("Invalid epsilon!")
            sys.exit(1)
        file1 = sys.argv[4]
        file2 = sys.argv[5]
    else:
        max_iter = 300
        try:
            eps = float(sys.argv[2])
        except ValueError:
            print("Invalid epsilon!")
            sys.exit(1)
        file1 = sys.argv[3]
        file2 = sys.argv[4]

    if not (1 < k):
        print("Invalid number of clusters!")
        sys.exit(1)
    if not (1 < max_iter < 1000):
        print("Invalid maximum iteration!")
        sys.exit(1)
    if not (eps >= 0):
        print("Invalid epsilon!")
        sys.exit(1)

    return k, max_iter, eps, file1, file2

def read_data(file1, file2):
    try:
        df1 = pd.read_csv(file1, header=None)
        df2 = pd.read_csv(file2, header=None)
        
        merged_df = pd.merge(df1, df2, on=0,how="inner")
        merged_df = merged_df.sort_values(by=0)
        
        indices = merged_df.iloc[:, 0].astype(int).tolist()
        
        data_points = merged_df.iloc[:, 1:].to_numpy()
        
        return data_points, indices
        
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


def kmeans_pp(data_points, k):
    np.random.seed(1234)
    
    n_samples, n_features = data_points.shape
    centroids = np.empty((k, n_features))
    centroid_indices = []

    # 1. Choose one center uniformly at random among the data points.
    first_idx = np.random.choice(n_samples)
    centroids[0] = data_points[first_idx]
    centroid_indices.append(first_idx)

    # 2. Initialize D(x) = distance to the first centroid
    distances = np.linalg.norm(data_points - centroids[0], axis=1)

    for i in range(1, k):
        # 2′. Update D(x) = min(previous D(x), distance to the new centroid)
        new_dists = np.linalg.norm(data_points - centroids[i-1], axis=1)
        distances = np.minimum(distances, new_dists)

        # 3. Build a fresh probability distribution ∝ D(x)
        probs = distances / distances.sum()

        #    and sample the next centroid index
        next_idx = np.random.choice(n_samples, p=probs)
        centroids[i] = data_points[next_idx]
        centroid_indices.append(next_idx)

    return centroids.tolist(), centroid_indices



def main():
    k, max_iter, eps, file1, file2 = parse_arguments()
    
    data_points, indices = read_data(file1, file2)
    
    if k >= len(data_points):
        print("Invalid number of clusters!")
        sys.exit(1)

    initial_centroids, initial_centroid_indices_from_data = kmeans_pp(data_points, k)
    
    # Convert data indices to original indices from the file
    initial_centroid_original_indices = [indices[i] for i in initial_centroid_indices_from_data]

    import mykmeanssp

    print(",".join(map(str, initial_centroid_original_indices)))

    final_centroids = mykmeanssp.fit(initial_centroids, data_points.tolist(), k, max_iter, eps, len(data_points), data_points.shape[1])
    
    for centroid in final_centroids:
        print(",".join([f"{c:.4f}" for c in centroid]))


if __name__ == "__main__":
    main()
    