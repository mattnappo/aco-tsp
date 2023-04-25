#ifndef __PAR_ACO_H__
#define __PAR_ACO_H__

// The output of an iteration
// Used to keep track of the best path seen so far
typedef struct ITER_T {
    int *path;    // Best path seen so far
    float length; // Associated path length
} iter_t;

// Print the iter_t structure
__host__ __device__ void print_iter(iter_t iter, int num_nodes);

__device__ int sample(int k, int *ints, float *weights);

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
__device__ void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

// Run a single ant, which will update tau and return an iter_t
__device__ iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t iter);

// Run the ant colony optimization algorithm on a graph.
__global__ void tour_construction(float *adjacency_matrix, float *A, int num_nodes,
        float a, float b, float p);

__global__ void pheromone_update(float *adjacency_matrix, float *tau, int num_nodes, int *tours, float a, float b, float p, int m);

#endif
