#!/bin/bash

make clean
make
#time ./cpu data/ts11.tsp sols/ts11.sol
#time ./gpu data/ts11.tsp sols/ts11.sol
FILE=data/dj38.tsp
SOLV=sols/dj38.sol
#./cpu     $FILE $SOLV
#./cpu_omp $FILE $SOLV
./gpu     $FILE $SOLV

# make && OMP_NUM_THREADS=16 ./cpu_omp data/dj38.tsp sols/dj38.sol
