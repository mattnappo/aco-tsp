#ifndef __ACO_H__
#define __ACO_H__

// The output of an iteration
typedef struct ITER_T {
    float *x_best; // Best path seen so far
    float  y_best; // Associated path length
} iter_t;

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b);

// Run a single ant, which will update tau and return an iter_t
iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t iter);

// Run the ant colony optimization algorithm on a graph.
void run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p);

#endif
