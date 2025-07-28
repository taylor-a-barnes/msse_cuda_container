import numpy as np
import time
import math
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule



cuda_module = SourceModule("""
__global__ void feedforward(
  int nlayers,
  int *sizes,
  float *activations,
  float *rawactivations,
  float *weights,
  float *biases) {

  int inode = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Offsets for the first hidden layer
  int node_offset_prev = 0;
  int node_offset = sizes[0];
  int biases_offset = 0;
  int weights_offset = 0;

  for (int ilayer = 1; ilayer < nlayers; ilayer++) {
    __syncthreads();
    int size_prev = sizes[ilayer-1];
    int size = sizes[ilayer];

    if ( inode < size ) {
      // Accumulate the biases and weights for this output value
      double sum = biases[ inode + biases_offset ];
      for (int jnode = 0; jnode < size_prev; jnode++) {
        double weight = weights[ (inode * size_prev) + jnode + weights_offset ];
        double input = activations[ jnode + node_offset_prev ];
        sum += weight * input;
      }
      rawactivations[ inode + node_offset ] = sum;
      activations[ inode + node_offset ] = sum;

      // Apply ReLU
      if ( ilayer < nlayers - 1 ) { // Don't apply ReLU to the output layer
        if ( activations[ inode + node_offset ] < 0.0f ) {
          activations[ inode + node_offset ] = 0.0f;
        }
      }
    }

    // Update the offsets
    node_offset_prev += size_prev;
    node_offset += size;
    biases_offset += size;
    weights_offset += size * size_prev;
  }

  return;
}


__global__ void backpropagation(
  int nlayers,
  int *sizes,
  float *activations,
  float *rawactivations,
  float *weights,
  float *biases,
  float *weights_grad,
  float *biases_grad,
  float *delta,
  float *reference) {

  int inode = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Offsets for the first hidden layer
  int node_offset_prev = 0;
  int node_offset = sizes[0];
  int biases_offset = 0;
  int weights_offset = 0;

  // Get the last offsets
  for (int ilayer = 1; ilayer < nlayers; ilayer++) {
    int size_prev = sizes[ilayer-1];
    int size = sizes[ilayer];
    node_offset_prev += size_prev;
    node_offset += size;
    biases_offset += size;
    weights_offset += size * size_prev;
  }

  for (int ilayer = nlayers-1; ilayer > 0; ilayer--) { // Iterate from the final layer to the first hidden layer
    __syncthreads();
    int size_prev = sizes[ilayer-1];
    int size = sizes[ilayer];

    // Update the offsets
    node_offset_prev -= size_prev;
    node_offset -= size;
    biases_offset -= size;
    weights_offset -= size * size_prev;

    if ( inode < size ) {

      if ( ilayer == nlayers - 1 ) { // Output layer
        delta[inode + biases_offset] = 2.0f * ( activations[inode + node_offset] - reference[inode] );
      }
      else { // Hidden layer
        float activation_grad = 0.0f;
        if ( rawactivations[inode + node_offset] > 0.0f ) {
          activation_grad = 1.0f;
        }
        delta[inode + biases_offset] = 0.0f;
        for ( int inext = 0; inext < sizes[ilayer+1]; inext++ ) {
          int weights_offset_next = weights_offset + (size * size_prev);
          int biases_offset_next = biases_offset + size;
          delta[inode + biases_offset] += weights[ (inext * size ) + inode + weights_offset_next ] * delta[ inext + biases_offset_next ];
        }
        delta[inode + biases_offset] *= activation_grad;
      }

      // Update the gradients
      biases_grad[inode + biases_offset] += delta[inode + biases_offset];
      for ( int iprev = 0; iprev < size_prev; iprev++ ) {
        int weights_offset_prev = weights_offset - (size * size_prev);
        weights_grad[ (inode * size_prev) + iprev + weights_offset] += delta[inode + biases_offset] * activations[iprev + node_offset_prev];
      }

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
backpropagation_gpu = cuda_module.get_function('backpropagation')



class InputLayer():
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros( (self.size,), dtype=np.float32 )

class Layer:
    def __init__(self, size, previous_layer):
        self.size = size
        self.previous_layer = previous_layer
        self.next_layer = None
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

    def backpropagation(self, reference):
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

        # Create data for the GPU
        # In order to avoid constantly sending data back and forth between the CPU and the GPU, it is beneficial to have all the weights, activations, of all layers stored together
        total_nodes = 0
        for layer in self.layers:
            total_nodes += layer.size
        self.activations = np.empty( total_nodes, dtype=np.float32 )
        self.activations_gpu = gpuarray.to_gpu( self.activations )
        self.rawactivations = np.empty( total_nodes, dtype=np.float32 )
        self.rawactivations_gpu = gpuarray.to_gpu( self.rawactivations )

        # Initialize the biases and the weights
        self.biases = np.empty( 0, dtype=np.float32 )
        self.weights = np.empty( 0, dtype=np.float32 )
        for ilayer in range( 1, len(self.layers) ):
            scale = np.sqrt( 2.0 / (self.layers[ilayer].size + self.layers[ilayer-1].size) )
            self.biases = np.append( self.biases, np.float32( np.random.uniform(-scale, scale, (self.layers[ilayer].size,)) ) )
            self.weights = np.append( self.weights, np.float32( np.random.uniform(-scale, scale, (self.layers[ilayer].size * self.layers[ilayer-1].size)) ) )
        self.biases_gpu = gpuarray.to_gpu( self.biases )
        self.weights_gpu = gpuarray.to_gpu( self.weights )
        self.biases_grad = np.zeros_like( self.biases, dtype=np.float32 )
        self.biases_grad_gpu = gpuarray.to_gpu( self.biases_grad )
        self.weights_grad = np.zeros_like( self.weights, dtype=np.float32 )
        self.weights_grad_gpu = gpuarray.to_gpu( self.weights_grad )
        self.deltas = np.empty_like( self.biases, dtype=np.float32 )
        self.deltas_gpu = gpuarray.to_gpu( self.deltas )

        # Initialize the layer sizes
        self.sizes = np.array( sizes, np.int32 )
        self.sizes_gpu = gpuarray.to_gpu( self.sizes )

    def train(self, nepochs):
        feedforward_time = 0.0
        backpropagation_time = 0.0
        for iepoch in range(nepochs):
            loss = 0.0
            indices = np.random.permutation(ninputs)
            inputs_shuffled = self.training_inputs[indices]
            refs_shuffled = self.training_references[indices]

            batch_size = 32
            for istart in range(0, ninputs, batch_size):
                iend = min(istart + batch_size, ninputs)

                for iref in range(istart, iend):
                    reference = np.array( refs_shuffled[iref], dtype=np.float32 )

                    ###############
                    # Convert the weights and biases to the GPU
                    ###############
                    biases_offset = 0
                    weights_offset = 0
                    for ilayer in range(1, len(self.sizes)):
                        layer = self.layers[ilayer]
                        for inode in range( layer.size ):
                            self.biases[inode + biases_offset] = layer.biases[inode]
                            self.biases_grad[inode + biases_offset] = layer.biases_grad[inode]
                            for jnode in range( self.layers[ilayer-1].size ):
                                self.weights[(inode * self.layers[ilayer-1].size) + jnode + weights_offset] = layer.weights[inode][jnode]
                                self.weights_grad[(inode * self.layers[ilayer-1].size) + jnode + weights_offset] = layer.weights_grad[inode][jnode]
                        biases_offset += layer.size
                        weights_offset += layer.size * self.layers[ilayer-1].size
                    self.biases_gpu = gpuarray.to_gpu( self.biases )
                    self.weights_gpu = gpuarray.to_gpu( self.weights )
                    self.biases_grad_gpu = gpuarray.to_gpu( self.biases_grad )
                    self.weights_grad_gpu = gpuarray.to_gpu( self.weights_grad )

                    # Feedforward through the other layers
                    start_time = time.time()
                    self.activations_gpu[0] = inputs_shuffled[iref]
                    feedforward_gpu(
                        np.int32( len(self.sizes) ),
                        self.sizes_gpu,
                        self.activations_gpu,
                        self.rawactivations_gpu,
                        self.weights_gpu,
                        self.biases_gpu,
                        block=(32,1,1),
                        grid=(1,1,1))

                    # Set the input layer
                    self.layers[0].activations[0] = inputs_shuffled[iref]

                    ##############
                    # Convert the activations and rawactivations to the CPU
                    ##############
                    self.activations = self.activations_gpu.get()
                    self.rawactivations = self.rawactivations_gpu.get()
                    nodes_offset = self.layers[0].size
                    for ilayer in range(1, len(self.sizes)):
                        layer = self.layers[ilayer]
                        for inode in range( layer.size ):
                            layer.activations[inode] = self.activations[inode + nodes_offset]
                            layer.rawactivations[inode] = self.rawactivations[inode + nodes_offset]
                        nodes_offset += layer.size

                    #for ilayer in range( 1, len(self.layers) ):
                    #    self.layers[ilayer].feedforward()
                    loss += np.sum( (self.layers[-1].activations - reference)**2 )
                    feedforward_time += time.time() - start_time

                    # Do backpropagation
                    start_time = time.time()
                    #for ilayer in range( len(self.layers)-1 ):
                    #    self.layers[-1-ilayer].backpropagation(reference)

                    reference_gpu = gpuarray.to_gpu( reference )
                    backpropagation_gpu(
                        np.int32( len(self.sizes) ),
                        self.sizes_gpu,
                        self.activations_gpu,
                        self.rawactivations_gpu,
                        self.weights_gpu,
                        self.biases_gpu,
                        self.weights_grad_gpu,
                        self.biases_grad_gpu,
                        self.deltas_gpu,
                        reference_gpu,
                        block=(32,1,1),
                        grid=(1,1,1))

                    ###############
                    # Convert the weights and biases gradients to the CPU
                    ###############
                    self.weights_grad = self.weights_grad_gpu.get()
                    self.biases_grad = self.biases_grad_gpu.get()
                    biases_offset = 0
                    weights_offset = 0
                    for ilayer in range(1, len(self.sizes)):
                        layer = self.layers[ilayer]
                        for inode in range( layer.size ):
                            layer.biases_grad[inode] = self.biases_grad[inode + biases_offset]
                            for jnode in range( self.layers[ilayer-1].size ):
                                layer.weights_grad[inode][jnode] = self.weights_grad[(inode * self.layers[ilayer-1].size) + jnode + weights_offset]
                        biases_offset += layer.size
                        weights_offset += layer.size * self.layers[ilayer-1].size

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
rvalues = np.random.uniform(min_rvalue, max_rvalue, (ninputs,)).astype(np.float32)

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
#net.test()


