#include <iostream>
#include <cmath>
#include <cuda.h>

#include "graph.cuh"
#include "aco.cuh"

#define NUM_ANTS  10000000
#define NUM_ITER  10
#define ALPHA     1.0f
#define BETA      4.0f
#define RHO       0.5f

int main(int argc, char** argv) {

    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <filename> <solution.sol>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // parse the filename to get the number of nodes 
    int num_nodes = std::stoi(filename.substr(7, filename.find(".tsp") - 2));
    // filename = "data/" + filename;

    std::cout << "Number of nodes: " << num_nodes << std::endl;

    // make a node list from the x and y coordinates of the cities
    float *node_list = new float[num_nodes * 2];
    
    make_node_list(filename, node_list);
    // print_node_list(num_nodes, node_list);

    // make an adjacency matrix from the node list
    float *adjacency_matrix = new float[num_nodes * num_nodes];
    make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);
    // print_adjacency_matrix(num_nodes, adjacency_matrix);

    int n = num_nodes * num_nodes;
    float *tau = new float[n];
    float *eta = new float[n];
    float *A   = new float[n];

    // Initialize tau, eta, A
    float w;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            w = read_2D(adjacency_matrix, i, j, num_nodes);
            if (w != 0) {
                write_2D(eta, i, j, num_nodes, 1.0/w);
                write_2D(A, i, j, num_nodes, std::pow(1.0/w, RHO));
            } else {
                write_2D(eta, i, j, num_nodes, 0.0);
                write_2D(A, i, j, num_nodes, 0.0);
            }
            write_2D(tau, i, j, num_nodes, 1.0);
        }
    }

    /* -- kernel wrapper -- */

    // device copies of node list and adjacency matrix
    float *d_node_list;
    float *d_adjacency_matrix;
    float *d_tau;
    float *d_eta;
    float *d_A; // reward matrix A
    /* TODO
       lets think about tours. proposed idea: NxM matrix of ints where
       N = num nodes
       M = num ants
       Ants (threads) index into this big matrix to report their tour.
       Allows us to easily see all the tours at the end of each iteration
       Might be slow? -- no
       this is what we'll do
    */
    int *d_tours;
    bool *d_visited;
    float *d_tour_lengths;
    float *d_unvisited_attractiveness;
    int *d_neighbors;

    // TODO: check return codes of all cuda function calls

    // allocate memory on device
    cudaMalloc((void**)&d_node_list, num_nodes * 2 * sizeof(float)); // TODO I don't htink this needs to be on the GPU
    cudaMalloc((void**)&d_adjacency_matrix, n * sizeof(float));
    cudaMalloc((void**)&d_tau, n * sizeof(float));
    cudaMalloc((void**)&d_eta, n * sizeof(float));
    cudaMalloc((void**)&d_A, n * sizeof(float));
    cudaMalloc((void**)&d_tours, num_nodes * NUM_ANTS * sizeof(int));
    cudaMalloc((void**)&d_tour_lengths, NUM_ANTS * sizeof(float));
    cudaMalloc((void**)&d_visited, num_nodes * NUM_ANTS * sizeof(bool));
    cudaMalloc((void**)&d_unvisited_attractiveness, num_nodes * NUM_ANTS * sizeof(float));
    cudaMalloc((void**)&d_neighbors, num_nodes * NUM_ANTS * sizeof(int));

    
    // copy node list and adjacency matrix to device
    cudaMemcpy(d_node_list, node_list, num_nodes * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacency_matrix, adjacency_matrix, num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eta, eta, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, tau, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);

    //cudaMemset(d_tau,   0, n * sizeof(float));
    //cudaMemset(d_eta,   0, n * sizeof(float));
    //cudaMemset(d_A,     0, n * sizeof(float)); // Initially, it is just adj_mat
    cudaMemset(d_tours, 0, num_nodes * NUM_ANTS * sizeof(int));
    cudaMemset(d_tour_lengths, 0, NUM_ANTS * sizeof(float));
    cudaMemset(d_visited, 0, num_nodes * NUM_ANTS * sizeof(bool));
    cudaMemset(d_unvisited_attractiveness, 0, num_nodes * NUM_ANTS * sizeof(float));
    cudaMemset(d_neighbors, 0, num_nodes * NUM_ANTS * sizeof(int));

    // Run ACO tests
    int   m = NUM_ANTS;
    int   k = NUM_ITER;
    float a = ALPHA;
    float b = BETA;
    float p = RHO;

    int n_threads = 32; // warp size
    int n_blocks = m / n_threads;

    while (k >= 0) {
        // Perform ant tour construction
        std::cout << "Performing ant tour construction" << std::endl;
        tour_construction<<<1,1>>>(d_adjacency_matrix, d_A, num_nodes, d_tours, m, d_tour_lengths, d_visited, d_unvisited_attractiveness, d_neighbors);

        cudaDeviceSynchronize(); // Thread barrier

        // Update tau
        /* TODO: think about what this kernel actually needs
           To minimize data transfer, each ant can "return" its tour, and 1/l (where l is tour length)
           Then, the parallel pheromone update kernel can look at all the tours of all the ants
           and sum the 1/l's for each edge to deposit pheromones, updating tau accordingly.
           This kernel should also update the reward matrix A (A = tau^a * eta^b)
           NOTE: eta is just adj_mat

           General Q: What is the correct way to make a thread "return" a value?
           Does it write its output to some location in memory, or does it actually
           return a value? If the former, does the thread return a pointer to the memory
           it wrote to?
        */
        pheromone_update<<<1,1>>>(d_adjacency_matrix, d_A, d_tau, a, d_eta, b, num_nodes, d_tours, d_tour_lengths, m, p);

        k--;
    }

    // copy adjacency matrix back to host
    cudaMemcpy(adjacency_matrix, d_adjacency_matrix, num_nodes * num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // copy adjacency matrix back to host
    cudaMemcpy(adjacency_matrix, d_adjacency_matrix, num_nodes * num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    
    // print adjacency matrix
    // print_adjacency_matrix(num_nodes, adjacency_matrix);

    printf("run with m=%d k=%d a=%f b=%f p=%f\n",m,k,a,b,p);
    //print_iter(best, num_nodes);

    // Read optimal path
    std::vector<int> optimal = read_optimal(argv[2]);
    int *optimal_path = &optimal[0];
    float optimal_length = calc_path_length(num_nodes, adjacency_matrix, optimal_path, optimal.size());

    printf("optimal: \n");
    print_iter((iter_t) { optimal_path, optimal_length }, num_nodes);

    delete[] node_list;
    delete[] adjacency_matrix;

    cudaFree(d_node_list);
    cudaFree(d_adjacency_matrix);

    return 0;
}
