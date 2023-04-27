#include <iostream>
#include <limits>
#include <sys/time.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "graph.cuh"
#include "aco.cuh"

int main(int argc, char *argv[])
{
    struct aco_config config;
    if (parse_args(argc, argv, &config)) {
        return 1;
    }
    int num_nodes = config.num_nodes;

    // Make a node list and adj mat from the x and y coordinates of the cities
    float *node_list = new float[num_nodes * 2];
    make_node_list(config.filename, node_list);
    float *adjacency_matrix = new float[num_nodes * num_nodes];
    make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);

    // Run ACO tests
    int   m = config.num_ants;
    int   k = config.num_iter;
    float a = config.alpha;
    float b = config.beta;
    float p = config.rho;
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


    printf("%s run with m=%d k=%d a=%f b=%f p=%f\n",argv[0],m,k,a,b,p);
#ifdef USE_OMP
    printf("time: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
#else
    printf("time: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
#endif


    // Read optimal path
    std::vector<int> optimal = read_optimal(argv[2]);
    int *optimal_path = &optimal[0];
    float optimal_length = calc_path_length(num_nodes, adjacency_matrix, optimal_path, optimal.size());

    printf("error: %f\n", p_error(optimal_length, best.length));

    // Print paths
    if (config.debug) {
        print_iter(best, num_nodes);
        printf("optimal: \n");
        print_iter((iter_t) { optimal_path, optimal_length }, num_nodes);
    }

    delete[] node_list;
    delete[] adjacency_matrix;
}
