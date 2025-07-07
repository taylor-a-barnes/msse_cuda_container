import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule



# Create an initial array to do some work with
data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
data_gpu = gpuarray.to_gpu(data)

# Create an array to hold the output data
# "empty_like" creates an empty array of the same size and shape as another
result_gpu = gpuarray.empty_like(data_gpu)


# For reference, here was our ElementwiseKernel from before:
scalar_kernel = ElementwiseKernel(
"float *input, float *result",
"result[i] = (10 * input[i]) + 1;",
"scalar_kernel")




# Here is a custom kernel that does the same thing:
# Note: scalar_math2 is meant to be added later in the lesson
cuda_module = pycuda.compiler.SourceModule("""
__global__ void scalar_math(float *input, float *result) {
  result[threadIdx.x] = ( 10 * input[threadIdx.x] ) + 1;
}

__global__ void scalar_math2(float *input, float *result, float add) {
  result[threadIdx.x] = ( 10 * input[threadIdx.x] ) + add;
}
""")
scalar_math = cuda_module.get_function("scalar_math")



# Do the calculation on the GPU
# Note: We'll discuss the "block" argument later.  For now, just mention that the first value of the tuple is the number of elements we are looping over
scalar_math( data_gpu, result_gpu, block=(5,1,1) )

# Get the result back
result = result_gpu.get()
print(f"Result of scalar_math():  {result}")




# Note: Add scalar_math2 function to the module now
# Note: Mention that when passing singletons, you don't need to explicitly transfer to the GPU
# Do the calculation on the GPU
scalar_math2 = cuda_module.get_function("scalar_math2")
scalar_math2( data_gpu, result_gpu, np.float32(4), block=(5,1,1) )

# Get the result back
result = result_gpu.get()
print(f"Result of scalar_math2(): {result}")








