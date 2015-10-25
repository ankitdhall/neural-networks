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
y = np.array([ [0,1,0,1,0,1,0,1] ]).T

def run(X, y, iterations = 10000):
	#seed to randomly initialize synapse weights
	np.random.seed(7)

	#initialize sysnapse as a 3x1 vector with values [0,1]
	synapse0 = 2*np.random.random((3,1)) - 1

	#print synapse0
	layer1, layer0 = 0, 0
	for iteration in range(iterations):

		#layer0 is the input, has 3 neurons
		layer0 = X

		#let NN predict the value using current values of synapse0
		layer1 = sigmoid(np.dot(layer0, synapse0))

		#find out the difference between prediction and expected 'y' value
		error = y - layer1
		#print "Error:", error

		#calculate by how much should the synapse weights
		#be changed depending on the confidence of the NN
		delta = error * sigmoid(layer1, True)
		#print "Layer1:", layer1
		#print "sigmoid:", sigmoid(layer1, True)
		#print "Delta:", delta
		#print "\n\n"
		synapse0 = synapse0 + np.dot(layer0.T, delta)

#	print layer1, synapse0
	print error
<<<<<<< HEAD
run(X, y)
=======
run(X, y)
>>>>>>> 4d5ea62b9e9e85408f2975bf77997f6ff2705948
