#!/bin/bash

make clean
make
#time ./cpu data/ts11.tsp sols/ts11.sol
#time ./gpu data/ts11.tsp sols/ts11.sol

./cpu data/dj38.tsp sols/dj38.sol
./gpu data/dj38.tsp sols/dj38.sol
