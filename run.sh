#!/bin/bash

make clean
make
./cpu data/ts11.tsp sols/ts11.sol
./gpu data/ts11.tsp sols/ts11.sol
