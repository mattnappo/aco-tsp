#include <iostream>

#include "graph.hpp"
#include "aco.hpp"

#define SOLVE_FILE "/u/mnappo/ts10.sol"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // parse the filename to get the number of nodes 
    int num_nodes = std::stoi(filename.substr(2, filename.find(".tsp") - 2));
    filename = "data/" + filename;

    std::cout << "Number of nodes: " << num_nodes << std::endl;

    // make a node list from the x and y coordinates of the cities
    float *node_list = new float[num_nodes * 2];
    
    make_node_list(filename, node_list);
    print_node_list(num_nodes, node_list);

    // make an adjacency matrix from the node list
    float *adjacency_matrix = new float[num_nodes * num_nodes];
    make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);
    print_adjacency_matrix(num_nodes, adjacency_matrix);

    // Run ACO tests
    int   m = 100; // num ants
    int   k = 150; // num iter
    float a = 1.0; // alpha
    float b = 5.0; // beta
    float p = 0.5; // rho
    iter_t best = run_aco(adjacency_matrix, num_nodes, m, k, a, b, p);
    print_iter(best, num_nodes);

    // Read optimal path
    std::vector<int> optimal = read_optimal(SOLVE_FILE);
    int *optimal_path = &optimal[0];
    float optimal_length = calc_path_length(num_nodes, adjacency_matrix, optimal_path, optimal.size());

    printf("optimal: \n");
    print_iter((iter_t) { optimal_path, optimal_length }, num_nodes);

    print_node_list(num_nodes, node_list);

    delete[] node_list;
    delete[] adjacency_matrix;

}
