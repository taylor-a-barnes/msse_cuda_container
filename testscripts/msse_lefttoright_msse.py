import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math




def update_animation(frame):

    # Run an iteration of the "simulation" on the GPU
    blocksize = 32
    ngridx = math.ceil( image_width / blocksize )
    ngridy = math.ceil( image_height / blocksize )
    run_simulation(data_gpu,
                   data_new_gpu,
                   np.int32(image_width),
                   np.int32(image_height),
                   block=(blocksize,blocksize,1),
                   grid=(ngridx,ngridy,1))

    data_new = data_new_gpu.get()
    image_data = np.zeros( (image_height, image_width, 3), dtype=np.uint8 )
    for y in range(image_height):
        for x in range(image_width):
            if data_new[y][x] == 1:
                image_data[y][x] = [255, 255, 255]

    im.set_data(image_data)
    data_gpu[:] = data_new_gpu[:]
    return [im]



cuda_module = SourceModule("""
#include <stdint.h>

__global__ void run_simulation(uint8_t *data, uint8_t *data_new, int image_width, int image_height) {
  int x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int y = threadIdx.y + ( blockIdx.y * blockDim.y );

  if ( x < image_width && y < image_height ) {
    int index = ( y * image_width ) + x;

    int left_x = ( ( x - 1 ) + image_width ) % image_width;
    int left_value = data[ ( y * image_width ) + left_x ];
    data_new[index] = left_value;
  }
}
""")
run_simulation = cuda_module.get_function("run_simulation")



image_width = 500
image_height = 512
num_frames = 100

data = np.uint8( np.random.randint(2, size=(image_height, image_width)) )
data_gpu = gpuarray.to_gpu( data )

data_new = np.empty_like(data)
data_new_gpu = gpuarray.to_gpu(data_new)

fig, ax = plt.subplots()
im = ax.imshow( np.zeros((image_height, image_width, 3), dtype=np.uint8) )
ani = FuncAnimation(fig, update_animation, frames=num_frames)
ani.save("animation.gif", fps=10, dpi=200)
