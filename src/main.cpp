#include <iostream>

#include "graph.hpp"
#include "aco.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
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
    int   k = 100; // num iter
    float a = 1.0; // alpha
    float b = 5.0; // beta
    float p = 0.5; // rho
    run_aco(adjacency_matrix, num_nodes, m, k, a, b, p);
}
