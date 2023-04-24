#include <iostream>
#include <assert.h>
#include <fstream>

#include "graph.hpp"
#include "aco.hpp"

void test_sample()
{
    std::ofstream file;
    file.open("sample_test.txt");

    int ints[] = {3, 4, 5, 6, 9};

    float weights[] = { 0.2, 0.2, 0.1, 0.15, 0.32 };
    for (int i = 0; i < 100000; i++) {
        int x = sample(5, ints, weights);
        file << x << std::endl;
    }
    file.close();
}

int main(int argc, char *argv[])
{
    test_sample();

    return 0;
}
