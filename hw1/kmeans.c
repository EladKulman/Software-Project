#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEFAULT_ITER 400
#define EPSILON 0.001

/*
 * This program implements the k-means clustering algorithm in C.
 * It reads a dataset from standard input, parses command-line arguments,
 * and performs clustering to find the centroids of the clusters.
 */

/*
 * Structure representing a vector with its data and dimension.
 */
typedef struct
{
    double *data;  /* Array of data points */
    int dimension; /* Dimension of the vector */
} Vector;

/*
 * Structure representing a cluster with its vectors
 */
typedef struct
{
    Vector *data; /* Array of data vectors */
    int size;
    int capacity;
} VectorList;

/*
 * Structure representing a dataset containing multiple vectors.
 */
typedef struct
{
    Vector *vectors; /* Array of vectors */
    int count;       /* Number of vectors in the dataset */
    int dimension;   /* Dimension of the vectors in the dataset */
} Dataset;

/*
 * Parses command-line arguments to extract the number of clusters (K) and maximum iterations.
 *
 * Args:
 *   argc: Number of command-line arguments.
 *   argv: Array of command-line arguments.
 *   K: Pointer to store the number of clusters.
 *   iter: Pointer to store the maximum number of iterations.
 *
 * Returns:
 *   0 on success, 1 on failure.
 */
int parse_arguments(int argc, char *argv[], int *K, int *iter);

/*
 * Reads the dataset from standard input.
 *
 * Returns:
 *   Pointer to the dataset structure, or NULL on failure.
 */
Dataset *read_data(void);

/*
 * Frees the memory allocated for a dataset.
 *
 * Args:
 *   dataset: Pointer to the dataset to free.
 */
void free_dataset(Dataset *dataset);

/*
 * Calculates the Euclidean distance between two vectors.
 *
 * Args:
 *   v1: Pointer to the first vector.
 *   v2: Pointer to the second vector.
 *
 * Returns:
 *   The Euclidean distance between the two vectors.
 */
double euclidean_distance(Vector *v1, Vector *v2);

/*
 * Performs the k-means clustering algorithm.
 *
 * Args:
 *   dataset: Pointer to the dataset.
 *   K: Number of clusters.
 *   max_iter: Maximum number of iterations.
 */
void kmeans(Dataset *dataset, int K, int max_iter);

/*
 * Parses a line of input to create a vector.
 *
 * Args:
 *   line: Input line containing vector data.
 *   expected_dim: Expected dimension of the vector (-1 if unknown).
 *
 * Returns:
 *   Pointer to the created vector, or NULL on failure.
 */
Vector *parse_vector(char *line, int expected_dim);

/*
 * Frees the memory allocated for a vector.
 *
 * Args:
 *   vector: Pointer to the vector to free.
 */
void free_vector(Vector *vector);

int my_strlen(const char *str);

void append_to_cluster(VectorList *cluster, Vector *vec)
{
    if (cluster->size >= cluster->capacity)
    {
        cluster->capacity *= 2;
        cluster->data = realloc(cluster->data, cluster->capacity * sizeof(Vector));
    }
    cluster->data[cluster->size++] = *vec;
}

Dataset *create_sample_dataset(void) {
    Dataset *dataset;
    int i, j;
    double values[4][3] = {
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
        {100.0, 100.0, 100.0},
        {101.0, 101.0, 101.0}
    };

    dataset = (Dataset *)malloc(sizeof(Dataset));
    if (dataset == NULL) {
        printf("Allocation failed for dataset\n");
        exit(1);
    }

    dataset->count = 4;
    dataset->dimension = 3;

    dataset->vectors = (Vector *)malloc(dataset->count * sizeof(Vector));
    if (dataset->vectors == NULL) {
        printf("Allocation failed for vectors\n");
        free(dataset);
        exit(1);
    }

    for (i = 0; i < dataset->count; i++) {
        dataset->vectors[i].dimension = dataset->dimension;
        dataset->vectors[i].data = (double *)malloc(dataset->dimension * sizeof(double));

        if (dataset->vectors[i].data == NULL) {
            printf("Allocation failed for vector %d\n", i);
            /* Free all previously allocated memory */
            for (j = 0; j < i; j++) {
                free(dataset->vectors[j].data);
            }
            free(dataset->vectors);
            free(dataset);
            exit(1);
        }

        for (j = 0; j < dataset->dimension; j++) {
            dataset->vectors[i].data[j] = values[i][j];
        }
    }

    return dataset;
}

int main(void)
{
    int K, max_iter;
    Dataset *dataset;
    /*
    *  if (parse_arguments(argc, argv, &K, &max_iter) != 0)
    * {
    *    return 1;
    *}
    */   
    K = 2;
    max_iter = 200;
    dataset = create_sample_dataset();
    if (dataset == NULL)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    if (K <= 1 || K >= dataset->count)
    {
        printf("Incorrect number of clusters!\n");
        free_dataset(dataset);
        return 1;
    }
    kmeans(dataset, K, max_iter);
    free_dataset(dataset);
    return 0;
}

int parse_arguments(int argc, char *argv[], int *K, int *iter)
{
    if (argc != 2 && argc != 3)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }
    {
        char *endptr;
        double k_val = strtod(argv[1], &endptr);
        if (endptr == argv[1] || *endptr != '\0' || floor(k_val) != k_val)
        {
            printf("Incorrect number of clusters!\n");
            return 1;
        }
        *K = (int)k_val;
        if (*K < 2)
        {
            printf("Incorrect number of clusters!\n");
            return 1;
        }
    }
    if (argc == 3)
    {
        char *endptr;
        double it_val = strtod(argv[2], &endptr);
        if (endptr == argv[2] || *endptr != '\0' || floor(it_val) != it_val)
        {
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
        *iter = (int)it_val;
        if (*iter <= 1 || *iter >= 1000)
        {
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
    }
    else
    {
        *iter = DEFAULT_ITER;
    }
    return 0;
}

Dataset *read_data(void)
{
    Dataset *dataset;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int capacity = 100;
    int count = 0;
    int dimension = -1;
    Vector *temp_vector;

    dataset = malloc(sizeof(Dataset));
    if (dataset == NULL)
        return NULL;
    dataset->vectors = malloc(capacity * sizeof(Vector));
    if (dataset->vectors == NULL)
    {
        free(dataset);
        return NULL;
    }

    while ((read = getline(&line, &len, stdin)) != -1)
    {
        if (read <= 1)
            continue;
        if (line[read - 1] == '\n')
        {
            line[read - 1] = '\0';
        }
        temp_vector = parse_vector(line, dimension);
        if (temp_vector == NULL)
        {
            free(line);
            free_dataset(dataset);
            return NULL;
        }
        if (dimension == -1)
        {
            dimension = temp_vector->dimension;
        }
        if (temp_vector->dimension != dimension)
        {
            free_vector(temp_vector);
            free(line);
            free_dataset(dataset);
            return NULL;
        }
        if (count >= capacity)
        {
            capacity *= 2;
            dataset->vectors = realloc(dataset->vectors, capacity * sizeof(Vector));
            if (dataset->vectors == NULL)
            {
                free_vector(temp_vector);
                free(line);
                free_dataset(dataset);
                return NULL;
            }
        }
        dataset->vectors[count] = *temp_vector;
        free(temp_vector);
        count++;
    }

    free(line);
    dataset->count = count;
    dataset->dimension = dimension;

    if (count == 0)
    {
        free_dataset(dataset);
        return NULL;
    }
    return dataset;
}

Vector *parse_vector(char *line, int expected_dim)
{
    Vector *vector;
    int dimension = 0;
    int i, j, start, end;
    int line_len;

    line_len = my_strlen(line);
    for (i = 0; i < line_len; i++)
    {
        if (line[i] == ',')
            dimension++;
    }
    dimension++;
    if (dimension == 0)
        return NULL;
    if (expected_dim != -1 && dimension != expected_dim)
        return NULL;
    vector = malloc(sizeof(Vector));
    if (vector == NULL)
        return NULL;
    vector->data = malloc(dimension * sizeof(double));
    if (vector->data == NULL)
    {
        free(vector);
        return NULL;
    }
    vector->dimension = dimension;
    i = 0;
    start = 0;
    for (j = 0; j <= line_len && i < dimension; j++)
    {
        if (j == line_len || line[j] == ',')
        {
            end = j;
            line[end] = '\0';
            vector->data[i] = atof(&line[start]);
            start = j + 1;
            i++;
        }
    }
    return vector;
}

void free_vector(Vector *vector)
{
    if (vector)
    {
        if (vector->data)
            free(vector->data);
        free(vector);
    }
}

void free_dataset(Dataset *dataset)
{
    int i;
    if (dataset)
    {
        if (dataset->vectors)
        {
            for (i = 0; i < dataset->count; i++)
            {
                if (dataset->vectors[i].data)
                {
                    free(dataset->vectors[i].data);
                }
            }
            free(dataset->vectors);
        }
        free(dataset);
    }
}

double euclidean_distance(Vector *v1, Vector *v2)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < v1->dimension; i++)
    {
        double diff = v1->data[i] - v2->data[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void kmeans(Dataset *dataset, int K, int max_iter)
{
    Vector *centroids;
    Vector *new_centroids;
    VectorList *clusters;
    int i, j, k, iter;
    double max_change;
    int closest_cluster_index;
    double min_distance, distance;

    centroids = malloc(K * sizeof(Vector));
    new_centroids = malloc(K * sizeof(Vector));
    clusters = malloc(K * sizeof(VectorList));

    if (!centroids || !new_centroids || !clusters)
    {
        printf("An Error Has Occurred\n");
        return;
    }

    /* 1. Initialize centroids by selecting k first points from the dataset */
    for (k = 0; k < K; k++)
    {
        centroids[k].dimension = dataset->dimension;
        centroids[k].data = malloc(dataset->dimension * sizeof(double));
        new_centroids[k].dimension = dataset->dimension;
        new_centroids[k].data = malloc(dataset->dimension * sizeof(double));
        if (!centroids[k].data || !new_centroids[k].data)
        {
            printf("An Error Has Occurred\n");
            return;
        }
        for (j = 0; j < dataset->dimension; j++)
        {
            centroids[k].data[j] = dataset->vectors[k].data[j];
        }
    }

    for (iter = 0; iter < max_iter; iter++)
    {
        /* 2. Assign every point to the closest centroid */
        for (i = 0; i < K; i++)
        {
            clusters[i].size = 0;
            clusters[i].capacity = 4;
            clusters[i].data = malloc(clusters[i].capacity * sizeof(Vector));
        }

        for (i = 0; i < dataset->count; i++)
        {
            min_distance = euclidean_distance(&dataset->vectors[i], &centroids[0]);
            closest_cluster_index = 0;
            for (k = 1; k < K; k++)
            {
                distance = euclidean_distance(&dataset->vectors[i], &centroids[k]);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_cluster_index = k;
                }
            }
            append_to_cluster(&clusters[closest_cluster_index], &dataset->vectors[i]);
        }

        /* 2. Update centeroids */
        for (k = 0; k < K; k++)
        {
            for (j = 0; j < dataset->dimension; j++) {
                new_centroids[k].data[j] = 0.0;
            }
            for (i = 0; i < clusters[k].size; i++)
            {
                for (j = 0; j < dataset->dimension; j++)
                {
                    printf("%f ", clusters[k].data[i].data[j]);
                    new_centroids[k].data[j] += clusters[k].data[i].data[j];
                }
            }
            for (j = 0; j < dataset->dimension; j++)
            {
                new_centroids[k].data[j] /= clusters[k].size;
            }
        }

        for (k = 0; k < K; k++)
        {
            for (j = 0; j < dataset->dimension; j++)
            {
                printf("Value: %f\n", new_centroids[k].data[j]);
                centroids[k].data[j] = new_centroids[k].data[j];
            }
        }

        max_change = 0.0;
        for (k = 0; k < K; k++)
        {
            distance = euclidean_distance(&centroids[k], &new_centroids[k]);
            if (distance > max_change)
            {
                max_change = distance;
            }
        }

        if (max_change < EPSILON)
        {
            break;
        }
    }

    for (k = 0; k < K; k++)
    {
        for (j = 0; j < dataset->dimension; j++)
        {
            if (j > 0)
                printf(",");
            printf("%.4f", centroids[k].data[j]);
        }
        printf("\n");
    }

    for (k = 0; k < K; k++)
    {
        free(centroids[k].data);
        free(new_centroids[k].data);
    }
    free(centroids);
    free(new_centroids);    
}

int my_strlen(const char *str)
{
    int len = 0;
    while (str[len] != '\0')
        len++;
    return len;
}
