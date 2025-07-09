#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function to calculate Euclidean distance between two vectors
double euclidean_distance(double *v1, double *v2, int dim) {
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// The core K-means algorithm implementation in C
double* kmeans_c(double *initial_centroids, double *datapoints, int N, int d, int K, int max_iter, double eps) {
    double *centroids = (double*)malloc(K * d * sizeof(double));
    double *new_centroids = (double*)malloc(K * d * sizeof(double));
    int *cluster_sizes = (int*)calloc(K, sizeof(int));
    int *assignments = (int*)malloc(N * sizeof(int));
    int i, j, k, iter;

    if (!centroids || !new_centroids || !cluster_sizes || !assignments) {
        // Memory allocation failed
        if (centroids) free(centroids);
        if (new_centroids) free(new_centroids);
        if (cluster_sizes) free(cluster_sizes);
        if (assignments) free(assignments);
        return NULL;
    }

    // Initialize centroids with the provided initial_centroids
    for (i = 0; i < K * d; i++) {
        centroids[i] = initial_centroids[i];
    }

    for (iter = 0; iter < max_iter; iter++) {
        // Assign each data point to the closest centroid
        for (i = 0; i < N; i++) {
            double min_distance = -1.0;
            int closest_cluster = -1;
            for (k = 0; k < K; k++) {
                double distance = euclidean_distance(&datapoints[i * d], &centroids[k * d], d);
                if (closest_cluster == -1 || distance < min_distance) {
                    min_distance = distance;
                    closest_cluster = k;
                }
            }
            assignments[i] = closest_cluster;
        }

        // Reset new_centroids and cluster_sizes
        for (k = 0; k < K; k++) {
            cluster_sizes[k] = 0;
            for (j = 0; j < d; j++) {
                new_centroids[k * d + j] = 0.0;
            }
        }

        // Calculate the sum of vectors for each cluster
        for (i = 0; i < N; i++) {
            k = assignments[i];
            cluster_sizes[k]++;
            for (j = 0; j < d; j++) {
                new_centroids[k * d + j] += datapoints[i * d + j];
            }
        }

        // Calculate the new centroids
        for (k = 0; k < K; k++) {
            if (cluster_sizes[k] > 0) {
                for (j = 0; j < d; j++) {
                    new_centroids[k * d + j] /= cluster_sizes[k];
                }
            }
        }

        // Check for convergence
        double max_change = 0.0;
        for (k = 0; k < K; k++) {
            double distance = euclidean_distance(&centroids[k * d], &new_centroids[k * d], d);
            if (distance > max_change) {
                max_change = distance;
            }
        }

        // Update centroids
        for (i = 0; i < K * d; i++) {
            centroids[i] = new_centroids[i];
        }

        if (max_change < eps) {
            break;
        }
    }

    free(new_centroids);
    free(cluster_sizes);
    free(assignments);

    return centroids;
}

// Python wrapper for the C function
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *initial_centroids_py, *datapoints_py;
    int K, max_iter, N, d;
    double eps;

    // Parse arguments from Python
        if (!PyArg_ParseTuple(args, "OOiidii", &initial_centroids_py, &datapoints_py, &K, &max_iter, &eps, &N, &d)) {
        return NULL;
    }

    // Convert Python lists to C arrays
    double *initial_centroids = (double*)malloc(K * d * sizeof(double));
    double *datapoints = (double*)malloc(N * d * sizeof(double));

    if (!initial_centroids || !datapoints) {
        if(initial_centroids) free(initial_centroids);
        if(datapoints) free(datapoints);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for C arrays.");
        return NULL;
    }

    for (int i = 0; i < K; i++) {
        PyObject *centroid_row = PyList_GetItem(initial_centroids_py, i);
        for (int j = 0; j < d; j++) {
            PyObject *val = PyList_GetItem(centroid_row, j);
            initial_centroids[i * d + j] = PyFloat_AsDouble(val);
        }
    }

    for (int i = 0; i < N; i++) {
        PyObject *data_row = PyList_GetItem(datapoints_py, i);
        for (int j = 0; j < d; j++) {
            PyObject *val = PyList_GetItem(data_row, j);
            datapoints[i * d + j] = PyFloat_AsDouble(val);
        }
    }

    // Call the C K-means function
    double *final_centroids_c = kmeans_c(initial_centroids, datapoints, N, d, K, max_iter, eps);
    
    free(initial_centroids);
    free(datapoints);

    if (final_centroids_c == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "K-means algorithm failed in C.");
        return NULL;
    }

    // Convert the result back to a Python list
    PyObject *final_centroids_py = PyList_New(K);
    for (int i = 0; i < K; i++) {
        PyObject *row = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(final_centroids_c[i * d + j]));
        }
        PyList_SetItem(final_centroids_py, i, row);
    }

    free(final_centroids_c);

    return final_centroids_py;
}

// Method definition object
static PyMethodDef mykmeanssp_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Fits the K-means model."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef mykmeanssp_module = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "A C extension for K-means clustering.",
    -1,
    mykmeanssp_methods
};

// Module initialization function
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    return PyModule_Create(&mykmeanssp_module);
}
