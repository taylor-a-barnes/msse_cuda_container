import numpy as np
from time import time
import math



class InputLayer():
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros( (self.size,), dtype=np.float32 )


class Layer:
    def __init__(self, size, previous_layer = None):
        self.size = size
        self.previous_layer = previous_layer
        self.next_layer = None
        
        if self.previous_layer is not None:
            self.previous_layer.next_layer = self

        scale = 0.1
        
        
        # NOTE: For anything with more than one input, need to include that in the dimensions here
        self.weights = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size, self.previous_layer.size)) )
        self.biases = scale * np.float32( np.random.uniform(-1.0, 1.0, (self.size,)) )
        
        # The activations, before applying sigmoid
        self.rawactivations = np.zeros( (self.size,), dtype=np.float32 )
        
        self.activations = np.zeros( (self.size,), dtype=np.float32 )
        
        self.weights_grad = np.zeros( (self.size, self.previous_layer.size), dtype=np.float32 )
        self.biases_grad = np.zeros( (self.size,), dtype=np.float32 )
        
        self.relu_flag = 1
        self.sigmoid_flag = 0
        self.batch_size = 32
        self.weight_flag = 0
        self.bias_flag = 0
        self.delta = 0.0

    def feedforward(self):
        self.rawactivations = np.dot( self.weights,
            self.previous_layer.activations ) + self.biases

        if self.next_layer is None: # Don't apply sigmoid to the output layer
            self.activations = self.rawactivations
        else: 
            # Apply the sigmoid function
            self.activations = 1.0 / ( 1.0 + np.exp(-self.rawactivations) )
            # Apply ReLU
            #self.activations = np.maximum(0, self.rawactivations)  # ReLU for hidden layers

    def backpropagation(self, reference=None):

        if reference is None:
            # Get this from differentiating the sigmoid
            activation_grad = self.activations * (1.0 - self.activations)
            #activation_grad = np.where(self.rawactivations > 0, 1.0, 0.0)  # Derivative of ReLU

            self.delta = np.dot(self.next_layer.weights.transpose(), self.next_layer.delta) * activation_grad
        else:
            # Get this from differentiating the cost function
            cost_grad = 2.0 * (self.activations - reference)

            #self.delta = cost_grad * activation_grad
            self.delta = cost_grad
            
        self.biases_grad += self.delta
        #self.weights_grad += np.dot(self.delta, self.previous_layer.activations.transpose())
        #self.weights_grad += np.dot(self.delta.reshape(1,1), self.previous_layer.activations.reshape(-1,1))
        for i, delt in enumerate(self.delta):
            for j, act in enumerate(self.previous_layer.activations):
                self.weights_grad[i][j] += delt * act
    
    def apply_gradient(self, batch_size, training_rate):
        #print(f"weights before: {self.weights}")
        self.weights = self.weights - ( training_rate / batch_size ) * self.weights_grad
        self.biases = self.biases - ( training_rate / batch_size ) * self.biases_grad
        #print(f"weights after: {self.weights}")
        
        # zero the gradient arrays
        self.weights_grad = np.zeros( (self.size, self.previous_layer.size), dtype=np.float32 )
        self.biases_grad = np.zeros( (self.size,), dtype=np.float32 )




def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner


ninputs = 100
De = 10.0
re = 1.0
a = 1.0
min_rvalue = 0.5
max_rvalue = 5.0
training_rate = 0.1

# Randomly generate a set of distances
rvalues = np.float32( np.random.uniform(min_rvalue, max_rvalue, (ninputs,)) )
#print(f"rvalues: {rvalues}")

# Generate the reference energies for each of these distances
erefs = np.empty( (ninputs,), dtype=np.float32 )
for idx, r in enumerate(rvalues):
    erefs[idx] = morse_potential(De, re, a, r)
print(f"erefs: {erefs}")

# Create the input layer
input_layer = InputLayer(1)

# Create the hidden layer
hidden_layer = Layer(16, input_layer)
hidden_layer2 = Layer(16, hidden_layer)

# Create the output layer
output_layer = Layer(1, hidden_layer2)




for iepoch in range(1000):
    batch_size = ninputs
    indices = np.random.permutation(batch_size)
    rvalues_shuffled = rvalues[indices]
    erefs_shuffled = erefs[indices]
    for iref in range(batch_size):
        reference = erefs_shuffled[iref]
        #reference = 10.0
        input_layer.activations[0] = ( rvalues_shuffled[iref] - min_rvalue ) / ( max_rvalue - min_rvalue )
        hidden_layer.feedforward()
        hidden_layer2.feedforward()
        output_layer.feedforward()
        print(f"Output, ref: {output_layer.activations}, {reference}")

        # Do backpropagation
        output_layer.backpropagation(reference)
        hidden_layer2.backpropagation()
        hidden_layer.backpropagation()

    output_layer.apply_gradient(batch_size, training_rate)
    hidden_layer2.apply_gradient(batch_size, training_rate)
    hidden_layer.apply_gradient(batch_size, training_rate)

