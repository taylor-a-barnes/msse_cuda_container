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
""")
run_layer = run_layer_module.get_function('run_layer')
