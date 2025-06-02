import sys
import math

EPS = 0.001            # convergence threshold
DEFAULT_ITER = 400     # default maximum iterations

NUM_CLUST_ERR = "Incorrect number of clusters!"
MAX_ITER_ERR = "Incorrect maximum iteration!"
GENERIC_ERR   = "An Error Has Occurred"


# ---------- helpers --------------------------------------------------------- #
def euclid_sq(p, q):
    return sum((pi - qi) ** 2 for pi, qi in zip(p, q))


def add_pts(p, q):
    return [pi + qi for pi, qi in zip(p, q)]


def div_pt(p, denom):
    return [pi / denom for pi in p]


# ---------- argument parsing ------------------------------------------------ #
def parse_args():
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



# ---------- input ----------------------------------------------------------- #
def read_dataset():
    data = []
    dim = None
    for raw in sys.stdin.read().splitlines():
        if raw.strip() == "":          # ignore expected extra empty row
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


# ---------- core algorithm -------------------------------------------------- #
def kmeans(data, k, max_iter):
    n = len(data)
    d = len(data[0])

    if k >= n:
        raise ValueError(NUM_CLUST_ERR)

    centroids = [data[i][:] for i in range(k)]  # first-k datapoints (deep copy)

    for _ in range(max_iter):
        # Assignment ---------------------------------------------------------
        clusters = [[] for _ in range(k)]
        for p in data:
            idx = min(range(k), key=lambda i: euclid_sq(p, centroids[i]))
            clusters[idx].append(p)

        # Update -------------------------------------------------------------
        new_centroids = []
        for idx, cluster in enumerate(clusters):
            if cluster:  # non-empty
                acc = [0.0] * d
                for p in cluster:
                    acc = add_pts(acc, p)
                new_centroids.append(div_pt(acc, len(cluster)))
            else:        # keep the old centroid if cluster is empty
                new_centroids.append(centroids[idx])

        # Convergence test ---------------------------------------------------
        deltas = [math.sqrt(euclid_sq(nc, oc))
                  for nc, oc in zip(new_centroids, centroids)]
        if max(deltas) < EPS:
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids


# ---------- output ---------------------------------------------------------- #
def fmt(c):
    return ",".join(f"{coord:.4f}" for coord in c)


def main():
    try:
        k, max_iter = parse_args()
        data        = read_dataset()
        centroids   = kmeans(data, k, max_iter)
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
