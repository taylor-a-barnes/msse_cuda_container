import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule


# Let's create a simple image file



# Matplotlib dependencies
import matplotlib
import matplotlib.pyplot as plt



image_width = 32
image_height = 32
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)


# Create an image on the CPU
for y in range(image_height):
    for x in range(image_width):
        # Make the image more red as we move right
        image_data[y][x][0] = 255 * ( x / image_width );
        # Make the image more green as we move down
        image_data[y][x][1] = 255 * ( y / image_height );
plt.imsave("output.png", image_data)



# Let's do something similar, but this time on the GPU
# Here, we'll start by just making the whole thing gray
image_width = 32
image_height = 32
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
image_data_gpu = gpuarray.to_gpu(image_data)

cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void create_image(uint8_t *image_data) {
  int starting_index = 3 * threadIdx.x;
  image_data[starting_index + 0] = 128;
  image_data[starting_index + 1] = 128;
  image_data[starting_index + 2] = 128;
}
""")
create_image = cuda_module.get_function("create_image")

create_image(image_data_gpu,
             block=(32*32,1,1),
             grid=(1,1,1))
test_data = image_data_gpu.get()
plt.imsave("output_gpu.png", test_data)



# Let's do something similar, but this time on the GPU
# Now, let's use both the x and y dimension of our block
image_width = 32
image_height = 32
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
image_data_gpu = gpuarray.to_gpu(image_data)

cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void create_image(uint8_t *image_data, int image_width, int image_height) {
  int pixel_index = (threadIdx.y * image_width) + threadIdx.x;
  int starting_index = 3 * pixel_index;
  image_data[starting_index + 0] = 128;
  image_data[starting_index + 1] = 128;
  image_data[starting_index + 2] = 128;
}
""")
create_image = cuda_module.get_function("create_image")

create_image(image_data_gpu,
             np.int32(image_width),
             np.int32(image_height),
             block=(32,32,1),
             grid=(1,1,1))
test_data = image_data_gpu.get()
plt.imsave("output_gpu.png", test_data)




# Let's do something similar, but this time on the GPU
# Now, let's reproduce the original image we generated on the CPU
image_width = 32
image_height = 32
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
image_data_gpu = gpuarray.to_gpu(image_data)

cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void create_image(uint8_t *image_data, int image_width, int image_height) {
  int pixel_index = (threadIdx.y * image_width) + threadIdx.x;
  int starting_index = 3 * pixel_index;
  image_data[starting_index + 0] = 255 * ( ((float) threadIdx.x) / image_width );
  image_data[starting_index + 1] = 255 * ( ((float) threadIdx.y) / image_height );
  image_data[starting_index + 2] = 0;
}
""")
create_image = cuda_module.get_function("create_image")

create_image(image_data_gpu,
             np.int32(image_width),
             np.int32(image_height),
             block=(32,32,1),
             grid=(1,1,1))
test_data = image_data_gpu.get()
plt.imsave("output_gpu.png", test_data)






# Let's do this on a larger image
# We can only have blocks of total size 1024, so we'll need a grid
# Note: this is just the first step
image_width = 64
image_height = 64
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
image_data_gpu = gpuarray.to_gpu(image_data)

cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void create_image(uint8_t *image_data, int image_width, int image_height) {
  int pixel_index = (threadIdx.y * image_width) + threadIdx.x;
  int starting_index = 3 * pixel_index;
  image_data[starting_index + 0] = 255 * ( ((float) threadIdx.x) / image_width );
  image_data[starting_index + 1] = 255 * ( ((float) threadIdx.y) / image_height );
  image_data[starting_index + 2] = 0;
}
""")
create_image = cuda_module.get_function("create_image")

create_image(image_data_gpu,
             np.int32(image_width),
             np.int32(image_height),
             block=(32,32,1),
             grid=(2,2,1))
test_data = image_data_gpu.get()
plt.imsave("output_block.png", test_data)





# Let's do this on a larger image
# We can only have blocks of total size 1024, so we'll need a grid
image_width = 512
image_height = 512
image_data = np.zeros((image_height,image_width,3), dtype = np.uint8)
image_data_gpu = gpuarray.to_gpu(image_data)

cuda_module = pycuda.compiler.SourceModule("""
#include <stdint.h>

__global__ void create_image(uint8_t *image_data, int image_width, int image_height) {
  int image_x = threadIdx.x + ( blockIdx.x * blockDim.x );
  int image_y = threadIdx.y + ( blockIdx.y * blockDim.y );

  int pixel_index = (image_y * image_width) + image_x;
  int starting_index = 3 * pixel_index;
  image_data[starting_index + 0] = 255 * ( ((float) image_x) / image_width );
  image_data[starting_index + 1] = 255 * ( ((float) image_y) / image_height );
  image_data[starting_index + 2] = 0;
}
""")
create_image = cuda_module.get_function("create_image")

create_image(image_data_gpu,
             np.int32(image_width),
             np.int32(image_height),
             block=(32,32,1),
             grid=(16,16,1))
test_data = image_data_gpu.get()
plt.imsave("output_block.png", test_data)





