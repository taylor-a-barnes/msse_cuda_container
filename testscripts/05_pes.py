import numpy as np
from time import time
import math

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
        self.weights = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size, self.previous_layer.size)) )
        self.biases = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size,)) )
        
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
        for i, delt in enumerate(self.delta):
            for j, act in enumerate(self.previous_layer.activations):
                self.weights_grad[i][j] += delt * act
    
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

    def train(self, nepochs):
        for iepoch in range(nepochs):
            indices = np.random.permutation(ninputs)
            inputs_shuffled = self.training_inputs[indices]
            refs_shuffled = self.training_references[indices]

            #batch_size = ninputs
            batch_size = 32
            for istart in range(0, ninputs, batch_size):
                iend = min(istart + batch_size, ninputs)

                for iref in range(istart, iend):
                    reference = refs_shuffled[iref]

                    # Set the input layer
                    self.layers[0].activations[0] = inputs_shuffled[iref]

                    # Feedforward through the other layers
                    for ilayer in range( 1, len(self.layers) ):
                        self.layers[ilayer].feedforward()
                    print(f"Output, ref: {self.layers[-1].activations}, {reference}")

                    # Do backpropagation
                    for ilayer in range( len(self.layers)-1 ):
                        self.layers[-1-ilayer].backpropagation(reference)

                for ilayer in range( 1, len(self.layers) ):
                    self.layers[ilayer].apply_gradient(batch_size, training_rate)

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
training_rate = 0.1

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
net.train( 300 )
net.test()


