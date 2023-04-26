/**
 * @file graph.cpp
 * @brief Graph class implementation
 * @details This file contains the implementation of the Graph class meant for Traveling Salesman Problem on a Euclidean Grid
 * 
 * @author  Aayush Poudel
*/

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>

#include "graph.cuh"

// write a value to a 2D array
void write_2D(float *array, int row, int col, int dim2, float value)
{
    array[row * dim2 + col] = value;
}

// read a value from a 2D array
float read_2D(float *array, int row, int col, int dim2)
{
    if (row > 1000 || col > 1000)
        printf("read at %d %d %d\n", row, col, dim2);
    return array[row * dim2 + col];
}

// make a node list from the x and y coordinates of the cities
int make_node_list(std::string filename, float *node_list)
{
    // after the line "NODE_COORD_SECTION", store the x and y coordinates of the cities in node_list
    // csv file format: city number, x coordinate, y coordinate

    int num_nodes = 0;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        if (line.find("NODE_COORD_SECTION") != std::string::npos)
        {
            break;
        }
    }

    // store the x and y coordinates of the cities in node_list
    int i = 0;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string word;
        iss >> word; // skip city number
        iss >> word; 
        float x = std::stof(word);
        iss >> word;
        float y = std::stof(word);
        write_2D(node_list, i, 0, 2, x);
        write_2D(node_list, i, 1, 2, y);
        i++;
    }

    return num_nodes;
}

// make an adjacency matrix from the node list
void make_adjacency_matrix(int num_nodes, float *node_list, float *adjacency_matrix)
{
    // for each pair of cities, calculate the distance between them and store it in the adjacency matrix
    // the distance between two cities is the Euclidean distance between them
    // the adjacency matrix is symmetric

    // dynamically allocate memory for adjacency_matrix

    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            float x1 = read_2D(node_list, i, 0, 2);
            float y1 = read_2D(node_list, i, 1, 2);
            float x2 = read_2D(node_list, j, 0, 2);
            float y2 = read_2D(node_list, j, 1, 2);
            //float distance = 1.0f/sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
            float distance = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
            write_2D(adjacency_matrix, i, j, num_nodes, distance);
            write_2D(adjacency_matrix, j, i, num_nodes, distance);
        }
    }

}

// get the neighbors of a node
void get_neighbors(int num_nodes, float *adjacency_matrix, int node, int *neighbors)
{
    // get the neighbors of a node
    // the neighbors of a node are the nodes that are connected to it by an edge
    // the neighbors of a node are stored in the neighbors array

    for (int i = 0; i < num_nodes; i++)
    {
        neighbors[i] = read_2D(adjacency_matrix, node, i, num_nodes);
    }
}

// get unvisited neighbors of a node
int get_unvisited_neighbors(int num_nodes, float *adjacency_matrix, int node, int *neighbors, bool *visited)
{
    // get the unvisited neighbors of a node
    // the unvisited neighbors of a node are the nodes that are connected to it by an edge and have not been visited yet
    // the unvisited neighbors of a node are stored in the neighbors array
    // if there are no unvisited neighbors, return -1. Otherwise, return the number of unvisited neighbors

    int num_unvisited_neighbors = 0;
    float inf = std::numeric_limits<float>::max();
    for (int i = 0; i < num_nodes; i++)
    {
        float w = read_2D(adjacency_matrix, node, i, num_nodes);
        if (w > 0 && w < inf && !visited[i]) 
        {
            neighbors[num_unvisited_neighbors] = i;
            num_unvisited_neighbors++;
        }
    }
    if (num_unvisited_neighbors == 0)
    {
        return -1;
    }
    else
    {
        return num_unvisited_neighbors;
    }
}

__host__ __device__ float calc_path_length(int num_nodes, float *adjacency_matrix, int *path, int path_size)
{
    // calculate the length of a path
    // the length of a path is the sum of the distances between the nodes in the path

    if (num_nodes != path_size) {
        // fprintf(stderr, "err in calc_path_len: num_nodes != path_size (this should not happen)\n");
        return -1;
    }

    float length = 0.0f;
    for (int i = 0; i < path_size; i++)
    {
        length += read_2D(adjacency_matrix, path[i], path[(i + 1) % path_size], num_nodes);
    }
    // length += read_2D(adjacency_matrix, path[path_size - 1], path[0], num_nodes); // I dont think we need this
    return length;
}

// print the node list
void print_node_list(int num_nodes, float *node_list)
{
    // print the x and y coordinates of the cities
    for (int i = 0; i < num_nodes; i++)
    {
        std::cout << "City " << i << ": (" << read_2D(node_list, i, 0, 2) << ", " << read_2D(node_list, i, 1, 2) << ")" << std::endl;
    }
}

// print the adjacency matrix
void print_adjacency_matrix(int num_nodes, float *adjacency_matrix)
{
    // print the adjacency matrix
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            std::cout << read_2D(adjacency_matrix, i, j, num_nodes) << " ";
        }
        std::cout << std::endl;
    }
}

void display_matrix(int num_nodes, float *adjacency_matrix, char *name)
{
    if (name)
        printf("%s:\n", name);

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            float v = read_2D(adjacency_matrix, i, j, num_nodes);
            printf("%8.2f ", v);
        }
        std::cout << std::endl;
    }
}
