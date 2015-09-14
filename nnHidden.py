"""
	Toy neural netowork in Python
	Date: September 8, 2015
"""

import numpy as np

"""
	Sigmoid activation function for each 
"""
def sigmoid(arg, derivative = False):
	if derivative:
		return arg*(1-arg)
	else:
		return 1/(1+np.exp(-arg))

X = np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
y = np.array([ [0,1,1,0,1,0,0,1] ]).T

def run(X, y, iterations = 10000):
	#seed to randomly initialize synapse weights
	np.random.seed(7)

	#initialize sysnapse as a 3x4, 4x1 vector with values [0,1]
	synapse0 = 2*np.random.random((3,4)) - 1
	synapse1 = 2*np.random.random((4,1)) - 1

	#print synapse0
	layer2, layer1, layer0 = 0, 0, 0
	for iteration in range(iterations):

		#layer0 is the input, has 3 neurons
		layer0 = X

		#layer1 has 4 neurons
		layer1 = sigmoid(np.dot(layer0, synapse0))

		#layer2 is output layer, 1 neuron
		layer2 = sigmoid(np.dot(layer1, synapse1))


		#find out the difference between predicted output and expected 'y' value
		layer2_error = y - layer2
		#print "Error:", error

		#calculate by how much should the synapse weights
		#be changed depending on the confidence of the NN
		layer2_delta = layer2_error * sigmoid(layer2, True)

		layer1_error = layer2_delta.dot(synapse1.T)

		layer1_delta = layer1_error * sigmoid(layer1, True)

		#print "Delta:", delta
		synapse1 = synapse1 + layer1.T.dot(layer2_delta)
		synapse0 = synapse0 + layer0.T.dot(layer1_delta)

	#print layer1, synapse0
	print layer2_error
run(X, y)
