#!/bin/bash

make clean
make

FILE=data/dj38.tsp
SOLV=sols/dj38.sol

export OMP_NUM_THREADS=16

./cpu     $FILE $SOLV
./cpu_omp $FILE $SOLV
./gpu     $FILE $SOLV

