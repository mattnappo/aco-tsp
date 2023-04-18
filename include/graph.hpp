/**
 * @file graph.hpp
 * @brief Graph class definition
 * @details This file contains the definition of the Graph class meant for Traveling Salesman Problem on a Euclidean Grid
 * 
 * @author  Aayush Poudel
 * 
*/

#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <string>

// nodes are represented by the x and y coordinates of the cities
// all edges are undirected with weight equal to the distance between the two cities
// the graph is represented by an adjacency matrix

// node list stores the x and y coordinates of the cities
// adjacency matrix stores the distance between the cities

// TODO: inline read_2D and write_2D, or make them a macro to reduce
// function overhead since these are extremely frequently called fns
void write_2D(float *array, int row, int col, int dim2, float value);
float read_2D(float *array, int row, int col, int dim2);

int make_node_list(std::string filename, float *node_list);
void make_adjacency_matrix(int num_nodes, float *node_list, float *adjacency_matrix);

void print_node_list(int num_nodes, float *node_list);
void print_adjacency_matrix(int num_nodes, float *adjacency_matrix);
void display_matrix(int num_nodes, float *adjacency_matrix, char *name);

void get_neighbors(int num_nodes, float *adjacency_matrix, int node, int *neighbors);
int get_unvisited_neighbors(int num_nodes, float *adjacency_matrix, int node, int *neighbors, bool *visited);

// TODO: inline this?
float calc_path_length(int num_nodes, float *adjacency_matrix, int *path);

#endif
