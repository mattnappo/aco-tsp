#include <iostream>
#include <assert.h>
#include <fstream>

#include "graph.hpp"
#include "aco.hpp"

void test_sample()
{
    std::ofstream file;
    file.open("sample_test.txt");

    float weights[] = { 0.2, 0.2, 0.1, 0.15, 0.32, 0.03 };
    for (int i = 0; i < 10000000; i++) {
        int x = sample(6, weights);
        file << x << std::endl;
    }
    file.close();
}

int main(int argc, char *argv[])
{
    test_sample();

    return 0;
}
