#ifndef __ACO_H__
#define __ACO_H__

// The output of an iteration
// Used to keep track of the best path seen so far
typedef struct ITER_T {
    int *path;    // Best path seen so far
    float length; // Associated path length
} iter_t;

// Print the iter_t structure
void print_iter(iter_t iter, int num_nodes);

// Sample an integer in the range [0,k) according to k weights
int sample(int k, float *weights);

// Sum an array of n floats
float sum_array(int n, float *values);

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

// Run a single ant, which will update tau and return an iter_t
iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t iter);

// Run the ant colony optimization algorithm on a graph.
iter_t run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p);

#endif
