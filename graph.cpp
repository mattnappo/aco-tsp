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

#include "graph.hpp"

// write a value to a 2D array
void write_2D(float *array, int row, int col, int dim2, float value)
{
    array[row * dim2 + col] = value;
}

// read a value from a 2D array
float read_2D(float *array, int row, int col, int dim2)
{
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
void get_unvisited_neighbors(int num_nodes, float *adjacency_matrix, int node, int *neighbors, bool *visited)
{
    // get the unvisited neighbors of a node
    // the unvisited neighbors of a node are the nodes that are connected to it by an edge and have not been visited yet
    // the unvisited neighbors of a node are stored in the neighbors array

    for (int i = 0; i < num_nodes; i++)
    {
        if (visited[i] == false)
        {
            neighbors[i] = read_2D(adjacency_matrix, node, i, num_nodes);
        }
    }
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




