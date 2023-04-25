#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "par_aco.cuh"

// Print the iter_t structure
__host__ __device__ void print_iter(iter_t iter, int num_nodes);

__device__ int sample(int k, int *ints, float *weights);

// Sum an array of n floats
__device__ float sum_array(int n, float *values);

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
__device__ void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

// Run a single ant, which will update tau and return an iter_t
__device__ iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t iter);

// Run the ant colony optimization algorithm on a graph.
__global__ iter_t ant_colony_optimization(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p);

__global__ void pheromone_update(float *tau, iter_t iter, int num_nodes, float rho);

