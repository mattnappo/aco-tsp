#include <cmath>

#include "aco.hpp"
#include "graph.hpp"

#include <cstring>

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
    int path_len = 1;
    int path[num_nodes]; // Start the hamiltonian cycle at node 0
    memset(path, 0, num_nodes);

    bool visited[num_nodes];
    memset(visited, false, num_nodes);

    // Try to visit every node
    while (path_len < num_nodes) {
        // Get the neighbors of the last node visited
        int i = path[path_len-1];
        int neighbors[num_nodes];
        int num_unvisited = get_unvisited_neighbors(num_nodes, adjacency_matrix,
                i, neighbors, visited);
        // If path complete or ant got stuck, then return
        if (num_unvisited == -1) {
            return (iter_t) {
                .x_best = iter.x_best,
                .y_best = iter.y_best
            };
        }

        // Sample the attractivenesses of the unvisited neighbors
        float *as;

        //todo

        // mark as visited




    }
}

// Run the ant colony optimization algorithm on a graph.
void run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p)
{

}
