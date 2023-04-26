#include <cstring>
#include <cmath>
#include <limits>
#include <random>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#include "aco.cuh"
#include "graph.cuh"

// Print the iter_t structure
__host__ __device__ void print_iter(iter_t iter, int num_nodes)
{
    printf("iter_t {\n\tpath = [ ");
    for (int i = 0; i < num_nodes; i++) {
        printf("%d ", iter.path[i]);
    }
    printf("]\n\t len = %f\n}\n", iter.length);
}

// From https://stackoverflow.com/questions/69873685/how-to-randomly-pick-element-from-an-array-with-different-probabilities-in-c
int sample(int k, int *ints, float *weights)
{
    std::mt19937 gen(std::random_device{}());

    std::vector<double> chances(weights, weights+k);
    std::vector<int> points(ints, ints+k);

    std::discrete_distribution<std::size_t> d{chances.begin(), chances.end()};

    auto sampled_value = points[d(gen)];
    return sampled_value;
}

__device__ int par_sample(int k, int *ints, float *weights, curandState_t* state){
    // cuda random number generator
    float r = curand_uniform(state);
    float sum = 0;
    for(int i = 0; i < k; i++){
            sum += weights[i];     
            if(r < sum){
                    return ints[i];
            }
    }
    return ints[k-1];
    
}


// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b)
{
    for (int i = 0; i < num_nodes; i++) {
        // TODO: could just be normal ij loop
        int neighbors[num_nodes];
        get_neighbors(num_nodes, adjacency_matrix, i, neighbors);
        for (int j = 0; j < num_nodes; j++) {
            float t = read_2D(tau, i, j, num_nodes);
            float e = read_2D(eta, i, j, num_nodes);
            float v = std::pow(t, a) * std::pow(e, b);
            write_2D(A, i, j, num_nodes, v);
        }
    }
    // display_matrix(num_nodes, A, "Attraction Matrix");
}

// Run a single ant, which will update tau
void run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A, iter_t* iter)
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
        i = path[path_size-1]; // Last node visited
        // Get unvisited neighbors of the last node visited
        num_unvisited = get_unvisited_neighbors(num_nodes, adjacency_matrix,
            i, neighbors, visited);
        // If path complete or ant got stuck, return
        if (num_unvisited == -1) {
            return;
        }
        //printf("unvisited (%d) neighbors of %d: [ ", num_unvisited, i);
        //for (int jj = 0; jj < num_unvisited; jj++) {
        //    printf("%d ", neighbors[jj]);
        //}
        //printf("] ");

        // Collect the attractivenesses of the unvisited neighbors
        float as[num_unvisited];
        for (int j = 0; j < num_unvisited; j++) {
            as[j] = read_2D(A, i, j, num_unvisited);
        }
        //printf("[ ");
        //for (int jj = 0; jj < num_unvisited; jj++) {
        //    printf("%f ", as[jj]);
        //}
        //printf("]\n");

        // Sample the distribution
        int choice = sample(num_unvisited, neighbors, as);
        //printf("picked %d\n", choice);
        int next_node = choice;
        path[path_size++] = next_node;

        // Mark as visited
        visited[i] = true;
        // printf("\n");
    }
    // now pathsize = numnodes = 11

    /*
    printf("visited:\n");
    for (int j = 0; j < num_nodes; j++) {
        printf("%d ", visited[j] ? 1 : 0);
    }
    printf("\n");
    printf("ant walked:\n");
    for (int j = 0; j < num_nodes; j++) {
        printf("%d ", path[j]);
    }
    printf("\n");
    */

    // Compute path length (path distance) by summing edge weights along the path
    //printf("path size: %d\n", path_size);
    //display_matrix(num_nodes, adjacency_matrix, "adj mat");
    //printf("path: [ ");
    //for (int jj = 0; jj < path_size; jj++) {
    //    printf("%d ", path[jj]);
    //}
    //printf("]\n");
    float path_length = calc_path_length(num_nodes, adjacency_matrix, path, path_size);
    float w = 1.0/path_length; // Amount of new pheramones on each edge of the path

    // Update tau
    int node_x, node_y;
    float tk;
    for (int i = 1; i < path_size; i++) {
        node_x = path[i-1];
        node_y = path[i];
        tk = read_2D(tau, node_x, node_y, num_nodes);
        // atomic for open mp
            
        write_2D(tau, node_x, node_y, num_nodes, tk + w);
    }

    // Update maxima if this ant's path is better
    if (path_length < iter->length) {
        iter->length = path_length;
        memcpy(iter->path, path, num_nodes*sizeof(int));
    }
    return;
    
}

// Run the ant colony optimization algorithm on a graph.
void run_aco(float *adjacency_matrix, int num_nodes, int m, int k_max,
        float a, float b, float p, iter_t * best)
{
    // Initialize everything
    int n = num_nodes*num_nodes;
    float *tau = new float[n];
    float *eta = new float[n];
    float w;

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            w = read_2D(adjacency_matrix, i, j, num_nodes);
            if(w != 0) {
                write_2D(eta, i, j, num_nodes, 1.0/w);
            }else{
                write_2D(eta, i, j, num_nodes, 0.0);
            }
            write_2D(tau, i, j, num_nodes, 1.0);
        }
    }

    display_matrix(num_nodes, adjacency_matrix, "matrix start");
    display_matrix(num_nodes, tau, "tau start");
    display_matrix(num_nodes, eta, "eta start");

    float v;
    float *A = new float[n];
    float rho_c = 1.0f-p;



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
        // #pragma omp parallel for
        for (int a = 0; a < m; a++) {
            //printf("main loop (%d, %d)\n", k, a); // crashes at 151, 0
            run_ant(adjacency_matrix, num_nodes, tau, A, best);
        }
    }
    display_matrix(num_nodes, tau, "final tau");

    delete[] tau;
    delete[] eta;
    delete[] A;

    // return;
}

__global__ void tour_construction(float *adj_mat, float* attractiveness, const int num_nodes, int *d_tours, int num_ants, float* d_tour_lengths) {
    
    // int ant_index = blockIdx.x * blockDim.x + threadIdx.x;

    // curandState_t state;
    // curand_init(clock64(), 0, 0, &state);

    // int path_size = 1;
    
    // int path[num_nodes] = {0}; // Start the hamiltonian cycle at node 0

    // bool visited[num_nodes] = {0};

    // // Try to visit every node
    // int i;
    // int neighbors[num_nodes];
    // int num_unvisited;
    // while (path_size < num_nodes) {
    //     i = path[path_size-1]; // Last node visited
    //     // Get unvisited neighbors of the last node visited
    //     num_unvisited = get_unvisited_neighbors(num_nodes, adj_mat,
    //         i, neighbors, visited);
    //     // If path complete or ant got stuck, return
    //     if (num_unvisited == -1) {
    //         // write path to d_tours
    //         for (int j = 0; j < path_size; j++) {
    //             d_tours[ant_index*num_nodes + j] = path[j];
    //         }
    //         return;
    //     }
    //     //printf("unvisited (%d) neighbors of %d: [ ", num_unvisited, i);
    //     //for (int jj = 0; jj < num_unvisited; jj++) {
    //     //    printf("%d ", neighbors[jj]);
    //     //}
    //     //printf("] ");

    //     // Collect the attractivenesses of the unvisited neighbors
    //     float as[num_unvisited];
    //     for (int j = 0; j < num_unvisited; j++) {
    //         as[j] = read_2D(attractiveness, i, j, num_unvisited);
    //     }
    //     //printf("[ ");
    //     //for (int jj = 0; jj < num_unvisited; jj++) {
    //     //    printf("%f ", as[jj]);
    //     //}
    //     //printf("]\n");

    //     // Sample the distribution
    //     int choice = sample(num_unvisited, neighbors, as);
    //     //printf("picked %d\n", choice);
    //     int next_node = choice;
    //     path[path_size++] = next_node;

    //     // Mark as visited
    //     visited[i] = true;
    //     // printf("\n");
    // }
    // // now pathsize = numnodes = 11

    // /*
    // printf("visited:\n");
    // for (int j = 0; j < num_nodes; j++) {
    //     printf("%d ", visited[j] ? 1 : 0);
    // }
    // printf("\n");
    // printf("ant walked:\n");
    // for (int j = 0; j < num_nodes; j++) {
    //     printf("%d ", path[j]);
    // }
    // printf("\n");
    // */

    // // Compute path length (path distance) by summing edge weights along the path
    // //printf("path size: %d\n", path_size);
    // //display_matrix(num_nodes, adjacency_matrix, "adj mat");
    // printf("path: [ ");
    // for (int jj = 0; jj < path_size; jj++) {
    //    printf("%d ", path[jj]);
    // }
    // printf("]\n");
    // float path_length = calc_path_length(num_nodes, adj_mat, path, path_size);

    // // update d_tour_length with path_length on ant_index
    
    // d_tour_lengths[ant_index] = path_length;

    
};

__global__ void pheromone_update(float *adj_mat, float *attractiveness, float* tau, float alpha, float *eta, float beta, int num_nodes, int *tours, float *tour_lengths, int num_ants, float rho)
{
    // Do pheromone evaporation
    //tau ij = (1-p) tau ij
    float v;
    float rho_c = 1.0f - rho;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            v = read_2D(tau, i, j, num_nodes);
            write_2D(tau, i, j, num_nodes, rho_c * v);
        }
    }

    // Deposit new pheromones
    
    // O(n) search through tours_lengths (TODO: I WANT TO DIE WE NEED TO OPTIMIZE THIS. Reduction is ans)
    float min = tour_lengths[0]; // Smallest tour length
    int opt = 0; // Index of smallest tour length
    for (int i = 1; i < num_ants; i++) {
        if (tour_lengths[i] < min) {
            min = tour_lengths[i];
            opt = i;
        }
    }

    // Deposit 1/min new pheromones onto each edge of the optimal path
    float new_pheromones = 1.0f / min;
    for (int i = 0; i < num_nodes; i++) {
        v = read_2DI(tours, opt, i, num_nodes);
        write_2DI(tours, opt, i, num_nodes, v + new_pheromones);
    }

    // Update attractivenesses (A = tau^a * eta^b)
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            float t = read_2D(tau, i, j, num_nodes);
            float e = read_2D(eta, i, j, num_nodes);
            v = powf(t, alpha) * powf(e, beta);
            write_2D(attractiveness, i, j, num_nodes, v);
        }
    }
};
