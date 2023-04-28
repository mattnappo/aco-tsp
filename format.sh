#!/bin/bash

find ./src/ ./include/ -iname *.cuh -o -iname *.cu | xargs clang-format --style=file -i
