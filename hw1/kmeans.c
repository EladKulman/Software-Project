#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEFAULT_ITER 400
#define EPSILON 0.001

typedef struct {
    double *data;
    int dimension;
} Vector;

typedef struct {
    Vector *vectors;
    int count;
    int dimension;
} Dataset;

/* Function prototypes */
int parse_arguments(int argc, char *argv[], int *K, int *iter);
Dataset* read_data(void);
void free_dataset(Dataset *dataset);
double euclidean_distance(Vector *v1, Vector *v2);
void kmeans(Dataset *dataset, int K, int max_iter);
Vector* parse_vector(char *line, int expected_dim);
void free_vector(Vector *vector);
int my_strlen(const char *str);
void my_strcpy(char *dest, const char *src);
char* my_strtok(char *str, const char *delim);

int main(int argc, char *argv[]) {
    int K, iter;
    Dataset *dataset;
    
    /* Parse command line arguments */
    if (parse_arguments(argc, argv, &K, &iter) != 0) {
        return 1;
    }
    
    /* Read data from stdin */
    dataset = read_data();
    if (dataset == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    /* Validate K */
    if (K <= 1 || K >= dataset->count) {
        printf("Incorrect number of clusters!\n");
        free_dataset(dataset);
        return 1;
    }
    
    /* Run K-means */
    kmeans(dataset, K, iter);
    
    /* Clean up */
    free_dataset(dataset);
    return 0;
}

int parse_arguments(int argc, char *argv[], int *K, int *iter)
{
    /* accept either 1 or 2 user-supplied arguments (program name + …) */
    if (argc != 2 && argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    /* ----------- parse K -------------------------------------------------- */
    {
        char   *endptr;
        double  k_val = strtod(argv[1], &endptr);

        /*  valid number, nothing extra afterwards, and an integer value?   */
        if (endptr == argv[1] || *endptr != '\0' || floor(k_val) != k_val) {
            printf("Incorrect number of clusters!\n");
            return 1;
        }
        *K = (int)k_val;

        /*  K must be at least 2 (python’s  k < 2  check)                  */
        if (*K < 2) {
            printf("Incorrect number of clusters!\n");
            return 1;
        }
    }

    /* ----------- parse max_iter ------------------------------------------ */
    if (argc == 3) {
        char   *endptr;
        double  it_val = strtod(argv[2], &endptr);

        if (endptr == argv[2] || *endptr != '\0' || floor(it_val) != it_val) {
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
        *iter = (int)it_val;

        /*  strict:   1 < iter < 1000   (--> allowed 2 … 999)              */
        if (*iter <= 1 || *iter >= 1000) {
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
    } else {
        *iter = DEFAULT_ITER;   /* 400 */
    }

    return 0;   /*  success */
}

Dataset* read_data(void) {
    Dataset *dataset;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int capacity = 100;
    int count = 0;
    int dimension = -1;
    Vector *temp_vector;
    
    dataset = malloc(sizeof(Dataset));
    if (dataset == NULL) return NULL;
    
    dataset->vectors = malloc(capacity * sizeof(Vector));
    if (dataset->vectors == NULL) {
        free(dataset);
        return NULL;
    }
    
    while ((read = getline(&line, &len, stdin)) != -1) {
        /* Skip empty lines */
        if (read <= 1) continue;
        
        /* Remove newline if present */
        if (line[read-1] == '\n') {
            line[read-1] = '\0';
        }
        
        /* Parse vector */
        temp_vector = parse_vector(line, dimension);
        if (temp_vector == NULL) {
            free(line);
            free_dataset(dataset);
            return NULL;
        }
        
        /* Set dimension on first valid vector */
        if (dimension == -1) {
            dimension = temp_vector->dimension;
        }
        
        /* Check dimension consistency */
        if (temp_vector->dimension != dimension) {
            free_vector(temp_vector);
            free(line);
            free_dataset(dataset);
            return NULL;
        }
        
        /* Expand capacity if needed */
        if (count >= capacity) {
            capacity *= 2;
            dataset->vectors = realloc(dataset->vectors, capacity * sizeof(Vector));
            if (dataset->vectors == NULL) {
                free_vector(temp_vector);
                free(line);
                free_dataset(dataset);
                return NULL;
            }
        }
        
        dataset->vectors[count] = *temp_vector;
        free(temp_vector); /* Free the struct but not the data */
        count++;
    }
    
    free(line);
    dataset->count = count;
    dataset->dimension = dimension;
    
    if (count == 0) {
        free_dataset(dataset);
        return NULL;
    }
    
    return dataset;
}

Vector* parse_vector(char *line, int expected_dim) {
    Vector *vector;
    
    int dimension = 0;
    int i, j, start, end;
    int line_len;
    
    line_len = my_strlen(line);
    
    /* Count dimensions first by counting commas + 1 */
    for (i = 0; i < line_len; i++) {
        if (line[i] == ',') dimension++;
    }
    dimension++; /* Add 1 for the last element */
    
    if (dimension == 0) return NULL;
    if (expected_dim != -1 && dimension != expected_dim) return NULL;
    
    /* Allocate vector */
    vector = malloc(sizeof(Vector));
    if (vector == NULL) return NULL;
    
    vector->data = malloc(dimension * sizeof(double));
    if (vector->data == NULL) {
        free(vector);
        return NULL;
    }
    vector->dimension = dimension;
    
    /* Parse values manually */
    i = 0;
    start = 0;
    for (j = 0; j <= line_len && i < dimension; j++) {
        if (j == line_len || line[j] == ',') {
            end = j;
            /* Extract substring and convert to double */
            line[end] = '\0';
            vector->data[i] = atof(&line[start]);
            start = j + 1;
            i++;
        }
    }
    
    return vector;
}

void free_vector(Vector *vector) {
    if (vector) {
        if (vector->data) free(vector->data);
        free(vector);
    }
}

void free_dataset(Dataset *dataset) {
    int i;
    if (dataset) {
        if (dataset->vectors) {
            for (i = 0; i < dataset->count; i++) {
                if (dataset->vectors[i].data) {
                    free(dataset->vectors[i].data);
                }
            }
            free(dataset->vectors);
        }
        free(dataset);
    }
}

double euclidean_distance(Vector *v1, Vector *v2) {
    double sum = 0.0;
    int i;
    
    for (i = 0; i < v1->dimension; i++) {
        double diff = v1->data[i] - v2->data[i];
        sum += diff * diff;
    }
    
    return sqrt(sum);
}

void kmeans(Dataset *dataset, int K, int max_iter) {
    Vector *centroids;
    Vector *new_centroids;
    int *assignments;
    int *cluster_sizes;
    int i, j, k, iter;
    double max_change;
    int closest_cluster;
    double min_distance, distance;
    
    /* Allocate memory */
    centroids = malloc(K * sizeof(Vector));
    new_centroids = malloc(K * sizeof(Vector));
    assignments = malloc(dataset->count * sizeof(int));
    cluster_sizes = malloc(K * sizeof(int));
    
    if (!centroids || !new_centroids || !assignments || !cluster_sizes) {
        printf("An Error Has Occurred\n");
        return;
    }
    
    /* Initialize centroids and allocate their data */
    for (k = 0; k < K; k++) {
        centroids[k].dimension = dataset->dimension;
        centroids[k].data = malloc(dataset->dimension * sizeof(double));
        new_centroids[k].dimension = dataset->dimension;
        new_centroids[k].data = malloc(dataset->dimension * sizeof(double));
        
        if (!centroids[k].data || !new_centroids[k].data) {
            printf("An Error Has Occurred\n");
            return;
        }
        
        /* Initialize centroids as first K datapoints */
        for (j = 0; j < dataset->dimension; j++) {
            centroids[k].data[j] = dataset->vectors[k].data[j];
        }
    }
    
    /* K-means iterations */
    for (iter = 0; iter < max_iter; iter++) {
        /* Assignment step */
        for (i = 0; i < dataset->count; i++) {
            min_distance = euclidean_distance(&dataset->vectors[i], &centroids[0]);
            closest_cluster = 0;
            
            for (k = 1; k < K; k++) {
                distance = euclidean_distance(&dataset->vectors[i], &centroids[k]);
                if (distance < min_distance) {
                    min_distance = distance;
                    closest_cluster = k;
                }
            }
            assignments[i] = closest_cluster;
        }
        
        /* Update step - calculate new centroids */
        for (k = 0; k < K; k++) {
            cluster_sizes[k] = 0;
            for (j = 0; j < dataset->dimension; j++) {
                new_centroids[k].data[j] = 0.0;
            }
        }
        
        for (i = 0; i < dataset->count; i++) {
            k = assignments[i];
            cluster_sizes[k]++;
            for (j = 0; j < dataset->dimension; j++) {
                new_centroids[k].data[j] += dataset->vectors[i].data[j];
            }
        }
        
        for (k = 0; k < K; k++) {
            if (cluster_sizes[k] > 0) {
                for (j = 0; j < dataset->dimension; j++) {
                    new_centroids[k].data[j] /= cluster_sizes[k];
                }
            }
        }
        
        /* Check convergence */
        max_change = 0.0;
        for (k = 0; k < K; k++) {
            distance = euclidean_distance(&centroids[k], &new_centroids[k]);
            if (distance > max_change) {
                max_change = distance;
            }
        }
        
        /* Copy new centroids to current centroids */
        for (k = 0; k < K; k++) {
            for (j = 0; j < dataset->dimension; j++) {
                centroids[k].data[j] = new_centroids[k].data[j];
            }
        }
        
        /* Check for convergence */
        if (max_change < EPSILON) {
            break;
        }
    }
    
    /* Output final centroids */
    for (k = 0; k < K; k++) {
        for (j = 0; j < dataset->dimension; j++) {
            if (j > 0) printf(",");
            printf("%.4f", centroids[k].data[j]);
        }
        printf("\n");
    }
    
    /* Free memory */
    for (k = 0; k < K; k++) {
        free(centroids[k].data);
        free(new_centroids[k].data);
    }
    free(centroids);
    free(new_centroids);
    free(assignments);
    free(cluster_sizes);
}

int my_strlen(const char *str) {
    int len = 0;
    while (str[len] != '\0') len++;
    return len;
}

void my_strcpy(char *dest, const char *src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}

char* my_strtok(char *str, const char *delim) {
    static char *saved_str = NULL;
    char *token_start;
    int i;
    
    if (str != NULL) {
        saved_str = str;
    }
    
    if (saved_str == NULL) return NULL;
    
    /* Skip leading delimiters */
    while (*saved_str != '\0') {
        for (i = 0; delim[i] != '\0'; i++) {
            if (*saved_str == delim[i]) break;
        }
        if (delim[i] == '\0') break;
        saved_str++;
    }
    
    if (*saved_str == '\0') {
        saved_str = NULL;
        return NULL;
    }
    
    token_start = saved_str;
    
    /* Find end of token */
    while (*saved_str != '\0') {
        for (i = 0; delim[i] != '\0'; i++) {
            if (*saved_str == delim[i]) {
                *saved_str = '\0';
                saved_str++;
                return token_start;
            }
        }
        saved_str++;
    }
    
    saved_str = NULL;
    return token_start;
}
