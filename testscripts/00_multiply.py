import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel










# Create a NumPy array of floats
data = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# Do some scalar math with the values on the GPU
data_gpu = gpuarray.to_gpu(data)
data_gpu = (10 * data_gpu) + 1

# Retreive the data from the GPU
data = data_gpu.get()

print(data)



import time

# Do some timings on a large array
datacount = 100000000
print(f"Size of bigdata: {( 4 * datacount ) / ( 1024 * 1024 * 1024 )} gigabytes")
bigdata = np.float32( np.random.rand( datacount ) )

# As a reference, do some scalar math with the values on the CPU
start_time = time.time()
bigresult = (10 * bigdata) + 1
end_time = time.time()
print(f"Time to calculate on the CPU: {end_time - start_time}")

# Copy the values to the GPU
start_time = time.time()
bigdata_gpu = gpuarray.to_gpu(bigdata)
end_time = time.time()
print(f"Time to transfer data from the CPU to the GPU: {end_time - start_time}")

# Now do some scalar math with the values on the GPU
start_time = time.time()
bigresult_gpu = (10 * bigdata_gpu) + 1
end_time = time.time()
print(f"Time to calculate on the GPU: {end_time - start_time}")








# Note: At this point, need to discuss Python map and reduce.

# Here is an approach that doesn't use map
data = [1, 2, 3, 4, 5]
result = [ (10 * x) + 1 for x in data ]
print(f"Result of non-map: {result}")

# Here is the same thing, using map
data = [1, 2, 3, 4, 5]
lambdafunc = lambda x : (10 * x) + 1
result = list( map( lambdafunc, data ) )
print(f"Result of map: {result}")

# Here is an approach for summing an array that doesn't use reduce
data = [1, 2, 3, 4, 5]
x = 0 # The sum we are forming
for y in data:
    x += y
print(f"Sum, without using reduce: {x}")

# Here is an approach that uses reduce
from functools import reduce
data = [1, 2, 3, 4, 5]
lambdafunc = lambda x, y : x + y
result = reduce( lambdafunc, data )
print(f"Sum, using reduce: {result}")









# Now, we're going to write some explicit code to do the above scalar operations
scalar_kernel = ElementwiseKernel(
"float *input, float *result",
"result[i] = (10 * input[i]) + 1;",
"scalar_kernel")

start_time = time.time()
scalar_kernel(bigdata_gpu, bigresult_gpu)
end_time = time.time()
print(f"Time to calculate using our scalar kernel: {end_time - start_time}")








# Do some timings on a large array
datacount = 100000000
print(f"Size of bigdata: {( 4 * 2 * datacount ) / ( 1024 * 1024 * 1024 )} gigabytes")
bigdata1 = np.float32( np.random.rand( datacount ) )
bigdata2 = np.float32( np.random.rand( datacount ) )

# As a reference, multiply the values on the CPU
start_time = time.time()
bigproduct = bigdata1 * bigdata2
end_time = time.time()
print(f"Time to multiply on the CPU: {end_time - start_time}")

# Copy the values to the GPU
start_time = time.time()
bigdata1_gpu = gpuarray.to_gpu(bigdata1)
bigdata2_gpu = gpuarray.to_gpu(bigdata2)
end_time = time.time()
print(f"Time to transfer data from the CPU to the GPU: {end_time - start_time}")

# Now multiply the values on the GPU
start_time = time.time()
bigproduct_gpu = bigdata1_gpu * bigdata2_gpu
end_time = time.time()
print(f"Time to multiply on the GPU: {end_time - start_time}")







