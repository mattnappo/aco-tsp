Tasks

Aayush
- [x] Write a minimal GPU-friendly graph library (use classes not structs)
   - [x] Implement adjacency matrices (linearized representation)
   - [x] Implement a structure to represent a path
   - [x] Auxiliary graph operations (get neighbors, calc path weight, etc...)
   - [x] Write dataloader for TSP dataset(s)

Matt
- [X] Implement generic ACO algorithm
- [X] Implement ACO for TSP on the CPU, sequentially
- [X] Write tests

GPU:
- [X] GPU Initialization / kernel wrapper
- [ ] Write Kernels
   - [ ] Tour construction
   - [ ] Parallel pheromone update
- [ ] Write GPU-safe sampling function
- [ ] Benchmark
   - [ ] Benchmark the CPU impl
   - [ ] Benchmark the real GPU impl

Now:
- [-] Implement ACO for TSP on the GPU, using parallelism
- [ ] Benchmark
   - [ ] Benchmark the CPU impl
   - [ ] Benchmark the real GPU impl
- [ ] Compare correctness (how are we going to do this? answer: comparing path lengths)
- [ ] Search the project for `TODO` since I added some potential stuff

Known problems
* CPU
  * Invalid reads on paths in `iter_t`s
  * Not reaching optimal on dj38 and above
  * Freeing heap alloc arrays with `delete[]` causes `path` in `iter_t` to be null
* Search the project for `TODO` since I added some potential stuff

