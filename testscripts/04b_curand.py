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



cuda_module = SourceModule(no_extern_c=True, source="""
#include <stdint.h>
#include <curand_kernel.h>

__device__ uint8_t get_value(uint8_t *data, int x, int y, int image_width, int image_height) {
  int wrapped_x = ( x + image_width ) % image_width;
  int wrapped_y = ( y + image_height ) % image_height;
  return data[ (wrapped_y * image_width) + wrapped_x ];
}

extern "C" {
__global__ void run_simulation(uint8_t *data, uint8_t *data_new, int image_width, int image_height) {
  int x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int y = threadIdx.y + ( blockIdx.y * blockDim.y );

  if ( x < image_width && y < image_height ) {
    int index = ( y * image_width ) + x;

    int nneighbors = 
      get_value(data, x - 1, y + 0, image_width, image_height) +
      get_value(data, x + 1, y + 0, image_width, image_height) +
      get_value(data, x - 1, y - 1, image_width, image_height) +
      get_value(data, x + 0, y - 1, image_width, image_height) +
      get_value(data, x + 1, y - 1, image_width, image_height) +
      get_value(data, x - 1, y + 1, image_width, image_height) +
      get_value(data, x + 0, y + 1, image_width, image_height) +
      get_value(data, x + 1, y + 1, image_width, image_height);

    data_new[index] = data[index];
    if ( nneighbors < 2 ) {
      data_new[index] = 0;
    }
    else if ( nneighbors > 3 ) {
      data_new[index] = 0;
    }
    else if ( nneighbors == 3 ) {
      data_new[index] = 1;
    }
  }
}

__global__ void initialize(uint8_t *data, int image_width, int image_height) {
  int x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int y = threadIdx.y + ( blockIdx.y * blockDim.y );
  if ( x >= image_width || y >= image_height ) return;

  int index = ( y * image_width ) + x;

  // NOT GOOD! 
  // Each thread will evaluate clock() independently.
  // With threads getting slightly different values from clock(), while also having slightly different indices, it is very possible two threads could have the same seed.
  // Instead, could pass the base seed in, and then add the thread index.
  unsigned long long seed = clock() + index;
  curandState cr_state;
  curand_init( seed, 0, 0, &cr_state );

  if ( curand_uniform(&cr_state) > 0.5f ) {
    data[index] = 1;
  }
  else {
    data[index] = 0;
  }
}

}

""")
run_simulation = cuda_module.get_function("run_simulation")
initialize = cuda_module.get_function("initialize")



image_width = 500
image_height = 512
num_frames = 100

data = np.zeros( (image_height, image_width), dtype=np.uint8 )
data_gpu = gpuarray.to_gpu( data )

data_new = data
data_new_gpu = gpuarray.to_gpu(data_new)

# Compute random noise on the GPU
blocksize = 32
ngridx = math.ceil( image_width / blocksize )
ngridy = math.ceil( image_height / blocksize )
initialize(data_gpu,
           np.int32(image_width),
           np.int32(image_height),
           block=(blocksize,blocksize,1),
           grid=(ngridx,ngridy,1))

fig, ax = plt.subplots()
im = ax.imshow( np.zeros((image_height, image_width, 3), dtype=np.uint8) )
ani = FuncAnimation(fig, update_animation, frames=num_frames)
ani.save("animation.gif", fps=10, dpi=200)
