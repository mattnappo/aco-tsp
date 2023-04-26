#!/bin/bash

make clean
make
time ./cpu data/ts11.tsp sols/ts11.sol
time ./gpu data/ts11.tsp sols/ts11.sol
