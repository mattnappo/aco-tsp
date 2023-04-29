#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#include "aco.cuh"
#include "graph.cuh"

int main(int argc, char **argv) {
  struct aco_config config;
  if (parse_args(argc, argv, &config)) {
    return 1;
  }
  int num_nodes = config.num_nodes;
  int m = config.num_ants;
  int k = config.num_iter;
  float a = config.alpha;
  float b = config.beta;
  float p = config.rho;

  // Make node list and adj mat
  float *node_list = new float[num_nodes * 2];
  make_node_list(config.filename, node_list);
  float *adjacency_matrix = new float[num_nodes * num_nodes];
  make_adjacency_matrix(num_nodes, node_list, adjacency_matrix);

  int n = num_nodes * num_nodes;
  float *tau = new float[n];
  float *eta = new float[n];
  float *A = new float[n];

  struct timeval tval_before, tval_after, tval_result;
  gettimeofday(&tval_before, NULL);

  // Initialize tau, eta, A
  float w;
  for (int i = 0; i < num_nodes; i++) {
    for (int j = 0; j < num_nodes; j++) {
      w = read_2D(adjacency_matrix, i, j, num_nodes);
      if (w != 0) {
        write_2D(eta, i, j, num_nodes, 1.0 / w);
        write_2D(A, i, j, num_nodes, std::pow(1.0 / w, p));
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
  int *d_tours;
  bool *d_visited;
  float *d_tour_lengths;
  float *d_unvisited_attractiveness;
  int *d_best_path; // Final optimal path to be extracted after all iterations
  int *d_neighbors;

  // TODO: check return codes of all cuda function calls

  // allocate memory on device
  cudaMalloc((void **)&d_node_list,
             num_nodes * 2 *
                 sizeof(float)); // TODO I don't htink this needs to be on the GPU
  cudaMalloc((void **)&d_adjacency_matrix, n * sizeof(float));
  cudaMalloc((void **)&d_tau, n * sizeof(float));
  cudaMalloc((void **)&d_eta, n * sizeof(float));
  cudaMalloc((void **)&d_A, n * sizeof(float));
  cudaMalloc((void **)&d_tours, num_nodes * m * sizeof(int));
  cudaMalloc((void **)&d_tour_lengths, m * sizeof(float));
  cudaMalloc((void **)&d_visited, num_nodes * m * sizeof(bool));
  cudaMalloc((void **)&d_unvisited_attractiveness, num_nodes * m * sizeof(float));
  cudaMalloc((void **)&d_neighbors, num_nodes * m * sizeof(int));
  cudaMalloc((void **)&d_best_path, num_nodes * sizeof(int));

  // copy node list and adjacency matrix to device
  cudaMemcpy(d_node_list, node_list, num_nodes * 2 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_adjacency_matrix, adjacency_matrix, num_nodes * num_nodes * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_eta, eta, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tau, tau, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);

  // cudaMemset(d_tau,   0, n * sizeof(float));
  // cudaMemset(d_eta,   0, n * sizeof(float));
  // cudaMemset(d_A,     0, n * sizeof(float)); // Initially, it is just adj_mat
  cudaMemset(d_tours, 0, num_nodes * m * sizeof(int));
  cudaMemset(d_tour_lengths, 0, m * sizeof(float));
  cudaMemset(d_visited, 0, num_nodes * m * sizeof(bool));
  cudaMemset(d_unvisited_attractiveness, 0, num_nodes * m * sizeof(float));
  cudaMemset(d_neighbors, 0, num_nodes * m * sizeof(int));
  cudaMemset(d_best_path, 0, num_nodes * sizeof(int));

  // Run ACO tests

  int n_threads = 32; // warp size
  int n_blocks = m / n_threads;

  while (k >= 0) {
    // Perform ant tour construction
    // std::cout << "Performing ant tour construction" << std::endl;
    tour_construction<<<n_blocks, n_threads>>>(d_adjacency_matrix, d_A, num_nodes,
                                               d_tours, m, d_tour_lengths, d_visited,
                                               d_unvisited_attractiveness, d_neighbors);

    cudaDeviceSynchronize(); // Thread barrier

    // Update tau
    /* TODO: think about what this kernel actually needs
       To minimize data transfer, each ant can "return" its tour, and 1/l (where l is tour
       length) Then, the parallel pheromone update kernel can look at all the tours of all
       the ants and sum the 1/l's for each edge to deposit pheromones, updating tau
       accordingly. This kernel should also update the reward matrix A (A = tau^a * eta^b)
       NOTE: eta is just adj_mat

       General Q: What is the correct way to make a thread "return" a value?
       Does it write its output to some location in memory, or does it actually
       return a value? If the former, does the thread return a pointer to the memory
       it wrote to?
    */
    pheromone_update<<<1, 1>>>(d_adjacency_matrix, d_A, d_tau, a, d_eta, b, num_nodes,
                               d_tours, d_tour_lengths, m, p, d_best_path);

    k--;
  }

  // Copy best path back to host
  int best_path[num_nodes];
  cudaMemcpy(best_path, d_best_path, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

  gettimeofday(&tval_after, NULL);
  timersub(&tval_after, &tval_before, &tval_result);

  float best_len = calc_path_length(num_nodes, adjacency_matrix, best_path, num_nodes);

  // Read optimal path
  std::vector<int> optimal = read_optimal(config.solution);
  int *optimal_path = &optimal[0];
  float optimal_length =
      calc_path_length(num_nodes, adjacency_matrix, optimal_path, optimal.size());

  printf("%s run with m=%d k=%d a=%f b=%f p=%f\n", argv[0], m, config.num_iter, a, b, p);
  printf("time: %ld.%06ld\n", (long int)tval_result.tv_sec,
         (long int)tval_result.tv_usec);
  printf("error: %f\n", p_error(optimal_length, best_len));

  if (config.debug) {
    printf("[ ");
    for (int i = 0; i < num_nodes; i++) {
      printf("%d ", best_path[i]);
    }
    printf("]\n");
    printf("len = %f\n", best_len);

    printf("optimal: \n");
    print_iter((iter_t){optimal_path, optimal_length}, num_nodes);
  }

  // Cleanup
  delete[] node_list;
  delete[] adjacency_matrix;

  delete[] tau;
  delete[] eta;
  delete[] A;

  cudaFree(d_node_list);
  cudaFree(d_adjacency_matrix);

  return 0;
}
