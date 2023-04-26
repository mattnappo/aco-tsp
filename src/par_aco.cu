#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "par_aco.cuh"

// Print the iter_t structure
__host__ __device__ void print_iter(iter_t iter, int num_nodes){
        printf("tour: [ ");
        for(int i = 0; i < num_nodes; i++){
                printf("%d ", iter.path[i]);
        }
        printf("]\n");
        printf("tour_length: %f\n", iter.length);
}

__device__ int sample(int k, int *ints, float *weights){
        // cuda random number generator
        curandState_t state;
        curand_init(clock64(), 0, 0, &state);
        float r = curand_uniform(&state);
        float sum = 0;
        for(int i = 0; i < k; i++){
                sum += weights[i];     
                if(r < sum){
                        return ints[i];
                }
        }
        return ints[k-1];
        
}

// Compute the edge attractiveness matrix given the graph, tau, eta, a, and b.
// Store the output in `float *A`
__device__ void edge_attractiveness(float *A, float *adjacency_matrix, int num_nodes,
        float *tau, float *eta, float a, float b){

        }

// Run a single ant, which will update tau and return an iter_t
__device__ iter_t run_ant(float *adjacency_matrix, int num_nodes, float *tau, float *A,
        iter_t iter);

// Run the ant colony optimization algorithm on a graph.
__global__ void tour_construction(float *adjacency_matrix, float *A, int num_nodes,
        float a, float b, float p){
                // test sample
                int k = 3;
                int ints[3] = {1, 2, 3};
                float weights[3] = {0.1, 0.2, 0.7};
                
                // counter array
                int counts[3] = {0, 0, 0};

                for (int i = 0; i < 1000; i++){
                        int s = sample(k, ints, weights);
                        counts[s-1] += 1;
                }
                printf("counts: [ ");
                for(int i = 0; i < k; i++){
                        printf("%d ", counts[i]);
                }
                printf("]\n");
                

        }

__global__ void pheromone_update(float *adjacency_matrix, float *tau, int num_nodes, int *tours, float a, float b, float p, int m){

}

