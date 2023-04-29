#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "graph.cuh"

static void str_split(std::string &str, std::vector<std::string> &out) {
  std::stringstream ss(str);
  std::string s;
  while (std::getline(ss, s, ' ')) {
    out.push_back(s);
  }
}

std::vector<int> read_optimal(std::string filename) {
  // Readlines of file
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open " << filename << " input file\n";
    return std::vector<int>();
  }

  // Skip first line
  std::string line;
  std::getline(file, line);

  // Read nodes
  // std::getline(file, line);
  std::stringstream joiner;
  while (std::getline(file, line)) {
    joiner << line;
  }
  joiner << " ";

  std::string nodes_str = joiner.str();
  // std::cout << nodes_str << std::endl; // Debug

  // Split and map to ints
  std::vector<std::string> nodes;
  str_split(nodes_str, nodes);
  nodes.pop_back();

  std::vector<int> int_nodes;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(int_nodes),
                 [](const std::string &str) { return std::stoi(str); });

  return int_nodes;
}

static void usage(char *name) {
  std::cout << "Usage: " << name
            << " <filename> <solution> <num ants> <num iter> <alpha> <beta> <rho>"
            << std::endl;
  std::cout << "\t<filename> - input file as a .tsp file" << std::endl;
  std::cout << "\t<solution> - solution file as a .sol file from the concorde solver"
            << std::endl;
  std::cout << "\t<num ants> - the number of ants" << std::endl;
  std::cout << "\t<num iter> - the number of iterations" << std::endl;
  std::cout << "\t<alpha> - tau (pheramone) weight" << std::endl;
  std::cout << "\t<beta> - eta (path length) weight" << std::endl;
  std::cout << "\t<rho> - pheromone decay rate" << std::endl;
  std::cout << "\t<debug> - print debug messages (true/false)" << std::endl << std::endl;
}

int find_int(const char *s) {
  int x;
  sscanf(s, "%*[^0123456789]%d", &x);
  return x;
}

int parse_args(int argc, char **args, struct aco_config *config) {
  if (argc != 9) {
    usage(args[0]);
    return 1;
  }

  // Basic config
  config->filename = args[1];
  config->solution = args[2];
  config->num_ants = atoi(args[3]);
  config->num_iter = atoi(args[4]);
  config->alpha = atof(args[5]);
  config->beta = atof(args[6]);
  config->rho = atof(args[7]);
  config->debug = (strcmp(args[8], "true") == 0) ? true : false;
  config->num_nodes = find_int(config->filename.c_str());

  return 0;
}

float p_error(float true_val, float obs_val) {
  return abs((obs_val - true_val) / true_val);
}

