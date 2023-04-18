#include <cstring>
#include <cmath>
#include <limits>

#include "aco.hpp"
#include "graph.hpp"

// Print the iter_t structure
void print_iter(iter_t iter, int num_nodes)
{
    printf("iter_t {\n\tpath = [ ");
    for (int i = 0; i < num_nodes; i++) {
        printf("%d ", iter.path[i]);
    }
    printf("]\n\t len = %f\n}\n", iter.length);
}

// Sum an array of n floats
// TODO: OpenMP this
float sum_array(int n, float *values)
{
    int i;
    float sum = 0.0;
    for (i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum;
}

// Sample an integer in the range [0,k) according to k weights
// TODO: Replace the linear search with binary search and OpenMP the loop
int sample(int k, float *weights)
{
    float sum = sum_array(k, weights);

    float p = (float)rand()/(float)(RAND_MAX/sum);
    float cum = 0;
    for (int i = 0; i < k; i++) {
        float w = weights[i];
        if (cum + w >= p) {
            return i;
        }
        cum += w;
    }
    return -1;
}

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b)
{
    for (int i = 0; i < num_nodes; i++) {
        int neighbors[num_nodes];
        get_neighbors(num_nodes, adjacency_matrix, i, neighbors);
        for (int j = 0; j < num_nodes; j++) {
            float t = read_2D(tau, i, j, num_nodes);
            float e = read_2D(eta, i, j, num_nodes);
            float v = std::pow(t, a) * std::pow(e, b);
            write_2D(A, i, j, num_nodes, v);
        }
    }
}

// Run a single ant, which will update tau
iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A, iter_t iter)
{
    // Initialize the ant's path
    int path_size = 1;
    int path[num_nodes]; // Start the hamiltonian cycle at node 0
    memset(path, 0, num_nodes);

    bool visited[num_nodes];
    memset(visited, 0, num_nodes);

    // Try to visit every node
    int i;
    int neighbors[num_nodes];
    int num_unvisited;
    while (path_size < num_nodes) {
        // Get the neighbors of the last node visited
        i = path[path_size-1];
        num_unvisited = get_unvisited_neighbors(num_nodes, adjacency_matrix,
            i, neighbors, visited);
        // If path complete or ant got stuck, return
        if (num_unvisited == -1) {
            return iter;
        }

        // Collect the attractivenesses of the unvisited neighbors
        float as[num_unvisited];
        for (int j = 0; j < num_unvisited; j++) {
            as[j] = read_2D(A, i, j, num_unvisited);
        }

        // Sample the distribution
        int next_node = neighbors[sample(num_unvisited, as)];
        path[path_size++] = next_node;

        // Mark as visited
        visited[i] = true;
    }

    // Compute path length (path distance) by summing edge weights along the path
    float path_length = calc_path_length(path_size, adjacency_matrix, path);
    float w = 1.0/path_length; // Amount of new pheramones on each edge of the path

    // Update tau
    // TODO: this probably has high cache miss rate
    int node_x, node_y;
    float tk;
    for (int i = 1; i < path_size; i++) {
        node_x = path[i-1];
        node_y = path[i];
        tk = read_2D(tau, node_x, node_y, num_nodes);
        write_2D(tau, node_x, node_y, num_nodes, tk + w);
    }

    // Update maxima if this ant's path is better
    if (path_length < iter.length) {
        return (iter_t) {
            .path = path,
            .length = path_length
        };
    } else {
        return iter;
    }
}

// Run the ant colony optimization algorithm on a graph.
iter_t run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p)
{
    // Initialize everything
    int n = num_nodes*num_nodes;
    float tau[n];
    float eta[n];
    float w;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            w = read_2D(adjacency_matrix, i, j, num_nodes);
            write_2D(eta, i, j, num_nodes, 1.0/w);
            write_2D(tau, i, j, num_nodes, 1.0);
        }
    }

    display_matrix(num_nodes, tau, "tau start");
    display_matrix(num_nodes, eta, "eta start");

    float v;
    float A[n] ={0.0};
    float rho_c = 1.0f-p;

    int best_path[num_nodes] = {0};
    float best_path_length = std::numeric_limits<float>::max();
    iter_t best = {
        .path = best_path,
        .length = best_path_length
    };

    // Main iteration loop (k-loop)
    for (int k = 0; k < k_max; k++) {
        // Compute reward matrix
        edge_attractiveness(A, adjacency_matrix, num_nodes, tau, eta, a, b);

        // Update tau decay factor
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_nodes; j++) {
                v = read_2D(tau, i, j, num_nodes);
                write_2D(tau, i, j, num_nodes, rho_c*v);
            }
        }

        // Run ants
        for (int a = 0; a < m; a++) {
            best = run_ant(adjacency_matrix, num_nodes, tau, A, best);
        }
        display_matrix(num_nodes, tau, "tau");
    }


    return best;
}
