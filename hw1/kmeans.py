import sys
import math

EPS = 0.001
DEFAULT_ITER = 400

NUM_CLUST_ERR = "Incorrect number of clusters!"
MAX_ITER_ERR = "Incorrect maximum iteration!"
GENERIC_ERR = "An Error Has Occurred"


def euclid_sq(p: list[float], q: list[float]) -> float:
    """
    Calculate the squared Euclidean distance between two points.

    Args:
        p (list of float): The first point.
        q (list of float): The second point.

    Returns:
        float: The squared Euclidean distance between p and q.
    """
    return sum((pi - qi) ** 2 for pi, qi in zip(p, q))


def add_pts(p: list[float], q: list[float]) -> list[float]:
    """
    Add two points element-wise.

    Args:
        p (list of float): The first point.
        q (list of float): The second point.

    Returns:
        list of float: The element-wise sum of p and q.
    """
    return [pi + qi for pi, qi in zip(p, q)]


def div_pt(p: list[float], denom: float) -> list[float]:
    """
    Divide each element of a point by a denominator.

    Args:
        p (list of float): The point.
        denom (float): The denominator.

    Returns:
        list of float: The point with each element divided by denom.
    """
    return [pi / denom for pi in p]


def parse_args() -> tuple[int, int]:
    """
    Parse command-line arguments for the number of clusters and maximum iterations.

    Returns:
        tuple: A tuple containing the number of clusters (k) and the maximum iterations (max_iter).

    Raises:
        ValueError: If the arguments are invalid.
    """
    if len(sys.argv) not in (2, 3):
        raise ValueError(GENERIC_ERR)

    try:
        k = float(sys.argv[1])
        if not k.is_integer():
            raise ValueError
        k = int(k)
    except ValueError:
        raise ValueError(NUM_CLUST_ERR)

    if len(sys.argv) == 3:
        try:
            iter_ = float(sys.argv[2])
            if not iter_.is_integer():
                raise ValueError
            iter_ = int(iter_)
        except ValueError:
            raise ValueError(MAX_ITER_ERR)
    else:
        iter_ = DEFAULT_ITER

    if not (1 < iter_ < 1000):
        raise ValueError(MAX_ITER_ERR)
    if k < 2:
        raise ValueError(NUM_CLUST_ERR)

    return k, iter_


def read_dataset() -> list[list[float]]:
    """
    Read and parse the dataset from standard input.

    Returns:
        list of list of float: The dataset as a list of points.

    Raises:
        ValueError: If the dataset is invalid.
    """
    data = []
    dim = None
    for raw in sys.stdin.read().splitlines():
        if raw.strip() == "":
            continue
        try:
            point = [float(x) for x in raw.strip().split(",")]
        except ValueError:
            raise ValueError(GENERIC_ERR)

        if dim is None:
            dim = len(point)
        elif len(point) != dim:
            raise ValueError(GENERIC_ERR)

        data.append(point)

    if not data:
        raise ValueError(GENERIC_ERR)
    return data


def kmeans(data: list[list[float]], k: int, max_iter: int) -> list[list[float]]:
    """
    Perform the k-means clustering algorithm.

    Args:
        data (list of list of float): The dataset.
        k (int): The number of clusters.
        max_iter (int): The maximum number of iterations.

    Returns:
        list of list of float: The final centroids of the clusters.

    Raises:
        ValueError: If the number of clusters is invalid.
    """
    n = len(data)
    d = len(data[0])

    if k >= n:
        raise ValueError(NUM_CLUST_ERR)

    #1. Initialize centroids by selecting k first points from the dataset
    centroids = [data[i][:] for i in range(k)]

    for _ in range(max_iter):
        #2. Assign every point to the closest centroid
        clusters = [[] for _ in range(k)]
        for p in data:
            idx = min(range(k), key=lambda i: euclid_sq(p, centroids[i]))
            clusters[idx].append(p)

        #3. Update centroids
        new_centroids = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                acc = [0.0] * d
                for p in cluster:
                    acc = add_pts(acc, p)
                new_centroids.append(div_pt(acc, len(cluster)))
            else:
                new_centroids.append(centroids[idx])

        deltas = [math.sqrt(euclid_sq(nc, oc))
                  for nc, oc in zip(new_centroids, centroids)]
        centroids = new_centroids

        # Update centroids until convergence or max iterations
        if max(deltas) < EPS:
            break


    return centroids


def fmt(c: list[float]) -> str:
    """
    Format a centroid for output.

    Args:
        c (list of float): The centroid.

    Returns:
        str: The formatted centroid as a comma-separated string with 4 decimal places.
    """
    return ",".join(f"{coord:.4f}" for coord in c)


def main() -> None:
    """
    Main function to execute the k-means clustering algorithm.

    Reads input, parses arguments, performs clustering, and prints the results.
    """
    try:
        k, max_iter = parse_args()
        data = read_dataset()
        centroids = kmeans(data, k, max_iter)
        for c in centroids:
            print(fmt(c))
    except ValueError as e:
        print(e)
        sys.exit(1)
    except Exception:
        print(GENERIC_ERR)
        sys.exit(1)


if __name__ == "__main__":
    main()
