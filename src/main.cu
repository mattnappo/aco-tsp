#include <iostream>
#include <cuda.h>

#include "graph.hpp"
#include "aco.hpp"
#include "par_aco.cuh"

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
    print_node_list(num_nodes, node_list);

    // make an adjacency matrix from the node list
    float *adjacency_matrix = new float[num_nodes * num_nodes];
    make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);
    print_adjacency_matrix(num_nodes, adjacency_matrix);

    // device copies of node list and adjacency matrix
    float *d_node_list;
    float *d_adjacency_matrix;

    // allocate memory on device
    cudaMalloc((void**)&d_node_list, num_nodes * 2 * sizeof(float));
    cudaMalloc((void**)&d_adjacency_matrix, num_nodes * num_nodes * sizeof(float));

    // copy node list and adjacency matrix to device
    cudaMemcpy(d_node_list, node_list, num_nodes * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacency_matrix, adjacency_matrix, num_nodes * num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run ACO tests
    int   m = 10000000; // num ants
    int   k = 10; // num iter
    float a = 1.0f; // alpha
    float b = 4.0f; // beta
    float p = .5; // rho

    int n_threads = 32; // warp size
    int n_blocks = m / n_threads;

    while(k >= 0) {
        // run ant colony optimization
        iter_t best_path = ant_colony_optimization<<<n_blocks, n_threads>>>(m, k, a, b, p, num_nodes, d_node_list, d_adjacency_matrix);
        cudaDeviceSynchronize();
        pheromone_update<<<1,1>>>(m, k, p, num_nodes, d_node_list, d_adjacency_matrix);        
        k--;
    }

    // copy adjacency matrix back to host
    cudaMemcpy(adjacency_matrix, d_adjacency_matrix, num_nodes * num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    
    // print adjacency matrix
    print_adjacency_matrix(num_nodes, adjacency_matrix);

    printf("run with m=%d k=%d a=%f b=%f p=%f\n",m,k,a,b,p);
    print_iter(best, num_nodes);

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