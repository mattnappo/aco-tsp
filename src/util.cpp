#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "graph.hpp"

void str_split(std::string &str, std::vector<std::string> &out) 
{ 
    std::stringstream ss(str);
    std::string s;
    while (std::getline(ss, s, ' ')) {
        out.push_back(s);
    }
}

std::vector<int> read_optimal(std::string filename)
{
    // Readlines of file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << " input file\n";
        return std::vector<int>();
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    // Remove first and last lines
    lines.erase(lines.begin());
    lines.erase(lines.end()-1);

    // Join into one space-separated line
    std::ostringstream joiner;
    for (const auto &l : lines) {
        joiner << l << " ";
    }
    std::string nodes_str = joiner.str();

    // Split and map to ints
    std::vector<std::string> nodes;
    str_split(nodes_str, nodes);
    nodes.pop_back();

    std::vector<int> int_nodes;
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(int_nodes),
        [](const std::string& str) {
            return std::stoi(str);
        });

    return int_nodes;
}

