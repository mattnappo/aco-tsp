#ifndef __ACO_H__
#define __ACO_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

// The output of an iteration
// Used to keep track of the best path seen so far
typedef struct ITER_T {
    int *path;    // Best path seen so far
    float length; // Associated path length
} iter_t;

// Print the iter_t structure
__host__ __device__ void print_iter(iter_t iter, int num_nodes);


int sample(int k, int *ints, float *weights);
__device__ int par_sample(int k, int *ints, float *weights, curandState_t *state);

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

__device__ void par_edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

// Run a single ant, which will update tau and return an iter_t
void run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t* iter);

// Run the ant colony optimization algorithm on a graph.
void run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p, iter_t *best);

__global__ void tour_construction(float *adj_mat, float* attractiveness, const int num_nodes, int *d_tours, int num_ants, float* d_tour_lengths, bool* d_visited, float* d_unvisited_attractiveness, int* d_neighbors);

__global__ void pheromone_update(
    float *adj_mat,
    float *attractiveness,
    float* tau,
    float alpha,
    float *eta,
    float beta,
    int num_nodes,
    int *tours,
    float *tour_lengths,
    int num_ants,
    float rho,
    int *best_path);

#endif
