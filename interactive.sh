#!/bin/sh

docker run --rm -it --gpus all -v $(pwd):/repo msse/cuda
