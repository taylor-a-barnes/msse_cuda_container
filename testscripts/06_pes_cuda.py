from __future__ import division
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np
from queue import Queue
import csv
from time import time
import math



run_layer_module = SourceModule("""
__global__ void run_layer(
  int n_in,
  float *in,
  int n_out,
  float *out,
  float *weights,
  float *biases,
  int relu_flag,
  int sigmoid_flag,
  int batch_size,
  int weight_flag,
  int bias_flag,
  float delta) {

  int i_out = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (i_out < n_out) {    // Prevents us from working outside the output vector 

    for (int i_batch = 0; i_batch < batch_size; i_batch++) {

      // Accumulate the bias and weights for this output value
      double sum = biases[ i_out ];
      for (int i_in = 0; i_in < n_in; i_in++) {
        double weight = weights[ (i_out * n_in) + i_in ];
        // Are we sure about this next line?  The indexing seems strange.
        double input = in[ (i_batch * n_in) + i_in ];
        // double input = in[ (i_batch * batch_size) + i_in ];
        sum += weight * input;
      }
      out[ (i_batch * n_out) + i_out] = sum;

    }

    // Compute the gradient with respect to the weights, if requested
    if ( weight_flag >= 0 && i_out == weight_flag / n_in ) {
      int i_in = weight_flag % n_in;
      for(int i_batch = 0; i_batch < batch_size; i_batch++) {
        out[ (i_batch * n_out) + i_out ] += delta * in[ (i_batch * n_in) + i_in ];
      }
    }

    // Compute the gradient with respect to the bias, if requested
    if ( bias_flag >= 0 && i_out == bias_flag ) {
      for (int i_batch = 0; i_batch < batch_size; i_batch++) {
        out[ (i_batch * n_out) + i_out ] += delta;
      }
    }

    // Apply ReLU, if requested
    if ( relu_flag > 0 ) {
      for ( int i_batch = 0; i_batch < batch_size; i_batch++) {
        if ( out[ (i_batch * n_out) + i_out ] < 0.0f ) {
          out[ (i_batch * n_out) + i_out ] = 0.0f;
        }
      }
    }

    // Apply sigmoid, if requested
    if ( sigmoid_flag > 0 ) {
      for ( int i_batch = 0; i_batch < batch_size; i_batch++) {
        float output = out[ (i_batch * n_out) + i_out ];
        out[ (i_batch * n_out) + i_out ] = 1.0f / ( 1.0f + expf(-output) );
      }
    }
  }
}

__global__ void backward(
  float *input,
  float *target,
  float *h,
  float *out,
  float *w1,
  float *w2,
  float *b1,
  float *b2) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if ( i > N_SAMPLES ) return;
    
    // Get the error in the final result
    float dL_dy = 2.0f * (out[i] - target[i]);

    // Update the weights for the output layer
    for (int j = 0; j < N_HIDDEN; ++j) {
        atomicAdd(&w2[j], -LEARNING_RATE * dL_dy * h[i * N_HIDDEN + j]);
        atomicAdd(&b2[0], -LEARNING_RATE * dL_dy);
    }

    // Update the weights for the hidden layer
    for (int j = 0; j < N_HIDDEN; ++j) {
        float dh = relu_deriv(h[i * N_HIDDEN + j]);
        for (int k = 0; k < N_INPUT; ++k) {
            atomicAdd(&w1[k * N_HIDDEN + j], -LEARNING_RATE * dL_dy * w2[j] * dh * input[i * N_INPUT + k]);
        }
        atomicAdd(&b1[j], -LEARNING_RATE * dL_dy * w2[j] * dh);
    }
}
""")
run_layer = run_layer_module.get_function('run_layer')



class InputLayer():
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros( (self.size,), dtype=np.float32 )
        self.activations_gpu = gpuarray.to_gpu( self.activations )


class Layer:
    def __init__(self, size, previous_layer = None):
        self.size = size
        self.previous_layer = previous_layer

        scale = 0.1
        
        
        # NOTE: For anything with more than one input, need to include that in the dimensions here
        self.weights = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.previous_layer.size, self.size)) )
        self.weights_gpu = gpuarray.to_gpu( self.weights )
        self.biases = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size,)) )
        self.biases_gpu = gpuarray.to_gpu( self.biases )
        
        self.activations = np.zeros( (size,), dtype=np.float32 )
        self.activations_gpu = gpuarray.to_gpu( self.activations )
        
        self.relu_flag = 1
        self.sigmoid_flag = 0
        self.batch_size = 32
        self.weight_flag = 0
        self.bias_flag = 0
        self.delta = 0.1

    def run(self):

        run_layer(
            np.int32( self.previous_layer.size ),
            self.previous_layer.activations_gpu,
            np.int32( self.size ),
            self.activations_gpu,
            self.weights_gpu,
            self.biases_gpu,
            np.int32( self.relu_flag ),
            np.int32( self.sigmoid_flag ),
            np.int32( self.batch_size ),
            np.int32( self.weight_flag ),
            np.int32( self.bias_flag ),
            np.float32( self.delta ),
            block=(32,1,1),
            grid=(1,1,1))
        self.activations = self.activations_gpu.get()



def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner

ninputs = 100
De = 1.0
re = 1.0
a = 1.0

# Randomly generate a set of distances
rvalues = np.float32( np.random.uniform(0.5, 5.0, (ninputs,)) )
#print(f"rvalues: {rvalues}")

# Generate the reference energies for each of these distances
erefs = np.empty( (ninputs,), dtype=np.float32 )
for idx, r in enumerate(rvalues):
    erefs[idx] = morse_potential(De, re, a, r)
print(f"erefs: {erefs}")

# Create the input layer
input = InputLayer(1)

# Create the hidden layer
hidden_layer = Layer(16, input)

# Create the output layer
output_layer = Layer(1, hidden_layer)

hidden_layer.run()
output_layer.run()
print(f"Output: {output_layer.values}")








