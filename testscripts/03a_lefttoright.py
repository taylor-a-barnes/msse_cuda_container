import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def update_animation(frame):

    # Run a single iteration of the "simulation" on the GPU
    blocksize = 32
    ngridx = math.ceil(image_width / blocksize)
    ngridy = math.ceil(image_height / blocksize)
    run_simulation(data_gpu,
                 np.int32(image_width),
                 np.int32(image_height),
                 block=(blocksize,blocksize,1),
                 grid=(ngridx,ngridy,1))

    # Output an image on the CPU
    data = data_gpu.get()
    image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
    for y in range(image_height):
        for x in range(image_width):
            if data[y][x] == 1:
                image_data[y][x] = [255,255,255]

    im.set_data(image_data)
    return [im]


cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void run_simulation(uint8_t *data, int image_width, int image_height) {
  int x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int y = threadIdx.y + ( blockIdx.y * blockDim.y );
  int index = (y * image_width) + x;

  int left_x = ( (x - 1) + image_width ) % image_width;
  int left_value = data[(y * image_width) + left_x];
  data[index] = left_value;
}
""")
run_simulation = cuda_module.get_function("run_simulation")


image_width = 512
image_height = 512
num_frames = 100

data = np.uint8( np.random.randint(2, size=(image_height, image_width)) )
# Do the following to emphasize the race condition:
#data[:][:] = np.arange(image_width) % 32
data_gpu = gpuarray.to_gpu(data)

# Animate the simulation, using the "update_animation" function
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((image_height, image_width, 3), dtype=np.uint8))
ani = FuncAnimation(fig,
                    update_animation,
                    frames=num_frames)
ani.save("animation.gif", fps=10, dpi=200)



