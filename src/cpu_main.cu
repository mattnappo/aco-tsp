#include <iostream>
#include <limits>
#include <time.h>

#include "graph.cuh"
#include "aco.cuh"

// #define SOLVE_FILE "./ts11.sol"

int main(int argc, char *argv[])
{
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

    // Run ACO tests
    int   m = 4096; // num ants
    int   k = 1000; // num iter
    float a = 1.0f; // alpha
    float b = 4.0f; // beta
    float p = .5; // rho
    int best_path[num_nodes] = {0};
    float best_path_length = std::numeric_limits<float>::max();
    iter_t best = {
        .path = best_path,
        .length = best_path_length
    };

    clock_t begin = clock();

    run_aco(adjacency_matrix, num_nodes, m, k, a, b, p, &best);

    clock_t end = clock();
    double dt = (double)(end - begin) / CLOCKS_PER_SEC;
#ifdef USE_OMP
    printf("ran cpu_omp in %f\n", dt);
#else
    printf("ran cpu in %f\n", dt);
#endif

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

}
