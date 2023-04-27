#include <iostream>
#include <limits>
#include <sys/time.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "graph.cuh"
#include "aco.cuh"

//#include "config.cuh"

int main(int argc, char *argv[])
{
#ifdef USE_OMP
    printf("using OpenMP\n");
#else
    printf("not using OpenMP\n");
#endif
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
    //print_node_list(num_nodes, node_list);

    // make an adjacency matrix from the node list
    float *adjacency_matrix = new float[num_nodes * num_nodes];
    make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);
    //print_adjacency_matrix(num_nodes, adjacency_matrix);

    // Run ACO tests
    int   m = NUM_ANTS;
    int   k = NUM_ITER;
    float a = ALPHA;
    float b = BETA;
    float p = RHO;
    int best_path[num_nodes] = {0};
    float best_path_length = std::numeric_limits<float>::max();
    iter_t best = {
        .path = best_path,
        .length = best_path_length
    };

    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);

    run_aco(adjacency_matrix, num_nodes, m, k, a, b, p, &best);

    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

#ifdef USE_OMP
    printf("ran cpu_omp in: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
#else
    printf("ran cpu in: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
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
