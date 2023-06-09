/**
 * @file graph.hpp
 * @brief Graph class definition
 * @details This file contains the definition of the Graph class meant for Traveling
 * Salesman Problem on a Euclidean Grid
 *
 * @author  Aayush Poudel
 * @author  Matt Nappo
 *
 */

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <string>
#include <vector>

// nodes are represented by the x and y coordinates of the cities
// all edges are undirected with weight equal to the distance between the two cities
// the graph is represented by an adjacency matrix

// node list stores the x and y coordinates of the cities
// adjacency matrix stores the distance between the cities

// TODO: inline read_2D and write_2D, or make them a macro to reduce
// function overhead since these are extremely frequently called fns
__host__ __device__ void write_2D(float *array, int row, int col, int dim2, float value);
__host__ __device__ float read_2D(float *array, int row, int col, int dim2);

__host__ __device__ void write_2DI(int *array, int row, int col, int dim2, int value);
__host__ __device__ int read_2DI(int *array, int row, int col, int dim2);

int make_node_list(std::string filename, float *node_list);
void make_adjacency_matrix(int num_nodes, float *node_list, float *adjacency_matrix);

void print_node_list(int num_nodes, float *node_list);
void print_adjacency_matrix(int num_nodes, float *adjacency_matrix);
void display_matrix(int num_nodes, float *adjacency_matrix, const char *name);

__host__ __device__ void get_neighbors(int num_nodes, float *adjacency_matrix, int node,
                                       int *neighbors);
__host__ __device__ int get_unvisited_neighbors(int num_nodes, float *adjacency_matrix,
                                                int node, int *neighbors, bool *visited);

// TODO: inline this?
__host__ __device__ float calc_path_length(int num_nodes, float *adjacency_matrix,
                                           int *path, int path_size);

struct aco_config {
  std::string filename;
  std::string solution;
  int num_nodes;
  int num_ants;
  int num_iter;
  float alpha;
  float beta;
  float rho;

  bool debug;
};

std::vector<int> read_optimal(std::string filename);
int parse_args(int argc, char **argv, struct aco_config *config);
float p_error(float true_val, float obs_val);

#endif
