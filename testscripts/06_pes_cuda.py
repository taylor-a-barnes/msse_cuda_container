import numpy as np
import time
import math
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule



cuda_module = SourceModule("""
__global__ void feedforward(
  int n_in,
  float *in,
  int n_out,
  float *out,
  float *raw,
  float *weights,
  float *biases) {

  int i_out = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (i_out >= n_out ) return;

  // Accumulate the biases and weights for this output value
  double sum = biases[ i_out ];
  for (int i_in = 0; i_in < n_in; i_in++) {
    double weight = weights[ (i_out * n_in) + i_in ];
    double input = in[ i_in ];
    sum += weight * input;
  }
  return;
  raw[ i_out ] = sum;
  out[ i_out ] = sum;

  // Compute the gradient with respect to the weights, if requested
  /*
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
  */

  // Apply ReLU
  if ( out[ i_out ] < 0.0f ) {
    out[ i_out ] = 0.0f;
  }
}


__global__ void backpropagation(
  int n_next,           // Number of nodes in the next layer
  float *delta_next,    // Delta values of nodes in the next layer
  int n_out,
  float *delta_out,
  float *activations,
  float *rawactivations,
  int n_prev,
  float *activations_prev,
  float *weights_next,
  float *reference,
  float *biases_grad,
  float *weights_grad) {

  if ( n_next <= 0 ) { // Output layer
    for ( int i_out = 0; i_out < n_out; i_out++) {
      delta_out[i_out] = 2.0f * ( activations[i_out] - reference[i_out] );
    }
  }
  else { // Hidden layer
    for ( int i_out = 0; i_out < n_out; i_out++ ) {
      float activation_grad = 0.0f;
      if ( rawactivations[i_out] > 0.0f ) {
        activation_grad = 1.0;
      }
      delta_out[i_out] = 0.0f;
      for ( int i_next = 0; i_next < n_next; i_next++ ) {
        // CHECK THIS LINE
        delta_out[i_out] = weights_next[ (i_next * n_out ) + i_out ] * delta_next[ (i_out * n_next) + i_next ];
      }
      delta_out[i_out] *= activation_grad;
    }
  }

  for ( int i_out = 0; i_out < n_out; i_out++ ) {
    biases_grad[i_out] += delta_out[i_out];
    for ( int i_prev = 0; i_prev < n_prev; i_prev++ ) {
      // CHECK THIS LINE
      weights_grad[ (i_out * n_prev) + i_prev] += delta_out[i_out] * activations_prev[i_prev];
    }
  }
}


__global__ void apply_gradient(
  int n_out,
  int n_prev,
  float* weights,
  float* biases,
  float* weights_grad,
  float* biases_grad,
  float training_rate,
  int batch_size
  ) {
  // Update the weights and biases
  for ( int i_out = 0; i_out < n_out; i_out++ ) {
    biases[i_out] -= ( training_rate / ( (float) batch_size) ) * biases_grad[i_out];
    for ( int i_prev = 0; i_prev < n_prev; i_prev++ ) {
      weights[ ( i_out * n_prev ) + i_prev ] -= ( training_rate / ( (float) batch_size) ) * weights_grad[ ( i_out * n_prev ) + i_prev ];
    }
  }

  // Zero the gradient arrays
  for ( int i_out = 0; i_out < n_out; i_out++ ) {
    biases_grad[i_out] = 0.0f;
    for ( int i_prev = 0; i_prev < n_prev; i_prev++ ) {
      weights_grad[ ( i_out * n_prev ) + i_prev ] = 0.0f;
    }
  }
}
""")
feedforward_gpu = cuda_module.get_function('feedforward')



class InputLayer():
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros( (self.size,), dtype=np.float32 )

class Layer:
    def __init__(self, size, previous_layer=None):
        self.size = size
        self.previous_layer = previous_layer
        self.next_layer = None
        if self.previous_layer is not None:
            self.previous_layer.next_layer = self

        # Xavier initialization
        scale = np.sqrt( 2.0 / (self.size + self.previous_layer.size) )
        self.biases = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size,)) )
        self.weights = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size, self.previous_layer.size)) )
        
        # The activations, before applying sigmoid
        self.rawactivations = np.zeros( (self.size,), dtype=np.float32 )
        self.activations = np.zeros( (self.size,), dtype=np.float32 )
        self.weights_grad = np.zeros( (self.size, self.previous_layer.size), dtype=np.float32 )
        self.biases_grad = np.zeros( (self.size,), dtype=np.float32 )
        self.delta = 0.0

    def feedforward(self):
        self.rawactivations = np.dot( self.weights,
            self.previous_layer.activations ) + self.biases

        if self.next_layer is None: # Just do linear output
            self.activations = self.rawactivations
        else:
            # Apply the activation function
            #self.activations = 1.0 / ( 1.0 + np.exp(-self.rawactivations) ) # Sigmoid
            self.activations = np.maximum(0, self.rawactivations) # ReLU
            #self.activations = np.tanh(self.rawactivations) # tanh

    def backpropagation(self, reference=None):
        if self.next_layer is None: # Output layer
            # Get this from differentiating the cost function
            self.delta = 2.0 * (self.activations - reference)
        else: # Hidden layer
            # Get this from differentiating the activation function
            #activation_grad = self.activations * (1.0 - self.activations) # Sigmoid
            activation_grad = np.where(self.rawactivations > 0, 1.0, 0.0) # ReLU
            #activation_grad = 1.0 - self.activations**2
            self.delta = np.dot(self.next_layer.weights.transpose(), self.next_layer.delta) * activation_grad

        self.biases_grad += self.delta
        self.weights_grad += np.outer(self.delta, self.previous_layer.activations)
    
    def apply_gradient(self, batch_size, training_rate):
        self.weights = self.weights - ( training_rate / batch_size ) * self.weights_grad
        self.biases = self.biases - ( training_rate / batch_size ) * self.biases_grad
        self.weights_grad.fill(0.0)
        self.biases_grad.fill(0.0)

class Network:
    def __init__(self, sizes, training_inputs, training_references):
        self.layers = []
        self.training_inputs = training_inputs
        self.training_references = training_references

        new_layer = InputLayer(1)
        self.layers.append( new_layer )
        for ilayer in range( 1, len(sizes) ):
            self.layers.append( Layer(sizes[ilayer], self.layers[-1]) )
            print(f"ilayer: {ilayer}")

        # Create data for the GPU
        # In order to avoid constantly sending data back and forth between the CPU and the GPU, it is beneficial to have all the weights, activations, of all layers stored together
        total_nodes = 0
        for layer in self.layers:
            total_nodes += layer.size
        self.activations = np.empty( total_nodes, dtype=np.float32 )
        self.activations_gpu = gpuarray.to_gpu( self.activations )
        self.rawactivations = np.empty( total_nodes, dtype=np.float32 )
        self.rawactivations_gpu = gpuarray.to_gpu( self.rawactivations )

        # Allocate space for the biases and the weights
        total_biases = 0
        total_weights = 0
        for ilayer in range( 1, len(self.layers) ):
            total_biases += self.layers[ilayer].size
            total_weights += self.layers[ilayer].size * self.layers[ilayer-1].size
        self.biases = np.empty( total_biases, dtype=np.float32 )
        self.weights = np.empty( total_weights, dtype=np.float32 )
        self.deltas = np.empty( total_biases, dtype=np.float32 )

        # Initialize the biases and the weights
        for ilayer in range( 1, len(self.layers) ):
            scale = np.sqrt( 2.0 / (self.layers[ilayer].size + self.layers[ilayer-1].size) )
            self.biases = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.layers[ilayer].size,)) )
            self.weights = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.layers[ilayer].size, self.layers[ilayer-1].size)) )
        self.biases_gpu = gpuarray.to_gpu( self.biases )
        self.weights_gpu = gpuarray.to_gpu( self.weights )

    def train(self, nepochs):
        feedforward_time = 0.0
        backpropagation_time = 0.0
        for iepoch in range(nepochs):
            loss = 0.0
            indices = np.random.permutation(ninputs)
            inputs_shuffled = self.training_inputs[indices]
            refs_shuffled = self.training_references[indices]

            #batch_size = ninputs
            batch_size = 32
            for istart in range(0, ninputs, batch_size):
                iend = min(istart + batch_size, ninputs)

                for iref in range(istart, iend):
                    reference = refs_shuffled[iref]

                    # Feedforward through the other layers
                    start_time = time.time()
                    self.activations_gpu[0] = inputs_shuffled[iref]
                    node_offset = 0
                    biases_offset = 0
                    weights_offset = 0
                    for ilayer in range( 1, len(self.layers) ):
                        n_in = np.float32( self.layers[ilayer-1].size )
                        inp = self.activations_gpu[node_offset:]
                        node_offset += self.layers[ilayer-1].size
                        n_out = np.float32( self.layers[ilayer].size )
                        out = self.activations_gpu[node_offset:]
                        raw = self.rawactivations_gpu[node_offset:]
                        b = self.biases_gpu[biases_offset:]
                        w = self.weights_gpu[weights_offset:]
                        feedforward_gpu(
                            n_in,
                            inp,
                            n_out,
                            out,
                            raw,
                            w,
                            b,
                            block=(32,1,1),
                            grid=(1,1,1))
                        biases_offset += self.layers[ilayer].size
                        weights_offset += self.layers[ilayer].size * self.layers[ilayer-1].size

                    # Set the input layer
                    self.layers[0].activations[0] = inputs_shuffled[iref]

                    for ilayer in range( 1, len(self.layers) ):
                        self.layers[ilayer].feedforward()
                    #print(f"Output, ref: {self.layers[-1].activations}, {reference}")
                    loss += np.sum( (self.layers[-1].activations - reference)**2 )
                    feedforward_time += time.time() - start_time

                    # Do backpropagation
                    start_time = time.time()
                    for ilayer in range( len(self.layers)-1 ):
                        self.layers[-1-ilayer].backpropagation(reference)
                    backpropagation_time += time.time() - start_time

                for ilayer in range( 1, len(self.layers) ):
                    self.layers[ilayer].apply_gradient(batch_size, training_rate)
            
            standard_deviation = math.sqrt( loss / ninputs )
            print(f"Epoch, deviation: {iepoch}, {standard_deviation}")
            
        print(f"Feedforward time: {feedforward_time}")
        print(f"Backpropagation time: {backpropagation_time}")


    def test(self):
        nvalues = 200
        rvalues = np.zeros( (ninputs,), dtype=np.float32 )
        erefs = np.zeros( (ninputs,), dtype=np.float32 )
        for ival in range(nvalues):
            rvalues[ival] = max_rvalue * ( ival/nvalues ) + min_rvalue * ( 1.0 - ival/nvalues )
            erefs[ival] = morse_potential(De, re, a, rvalues[ival])
            erefs[ival] = (erefs[ival] - mean_e) / std_e
            rvalues[ival] = ( rvalues[ival] - (max_rvalue + min_rvalue) / 2.0 ) / (max_rvalue - min_rvalue)

        for ival in range(nvalues):
            # Set the input layer
            self.layers[0].activations[0] = rvalues[ival]

            # Feedforward through the other layers
            for ilayer in range( 1, len(self.layers) ):
                self.layers[ilayer].feedforward()
            print(f"Testing, ref: {self.layers[-1].activations[0]}, {erefs[ival]}")


def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner


ninputs = 300
De = 1.0
re = 1.0
a = 1.0
min_rvalue = 0.5
max_rvalue = 2.0
training_rate = 0.02

# Randomly generate a set of distances
rvalues = np.float32( np.random.uniform(min_rvalue, max_rvalue, (ninputs,)) )

# Generate the reference energies for each of these distances
erefs = np.empty( (ninputs,), dtype=np.float32 )
for idx, r in enumerate(rvalues):
    erefs[idx] = morse_potential(De, re, a, r)
mean_e = np.mean(erefs)
std_e = np.std(erefs)
erefs_normalized = (erefs - mean_e) / std_e
print(f"erefs: {erefs}")
print(f"erefs_normalized: {erefs_normalized}")

rvalues_normalized = ( rvalues - (max_rvalue + min_rvalue) / 2.0 ) / (max_rvalue - min_rvalue)


net = Network( [1, 16, 16, 1], rvalues_normalized, erefs_normalized )
start_time = time.time()
net.train( 500 )
print(f"Training time: {time.time() - start_time}")
net.test()


