"""
	Neural network with back propagation
	24th October, 2015
"""
"""
TODO:
-random initialization
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z, derivative = False):
    
    if derivative:
        return z*(1-z)
    else:
        return 1./(1 + np.exp(-z))

def train(X, y, rate = 0.1, iterations = 100000):
    plotx, ploty = [], []    
    
    n0 = 3
    n1 = 4
    n2 = 4
    nOutput = 1

    synapse1 = np.random.rand(n1, n0)
    synapse2 = np.random.rand(n2, n1)
    synapse3 = np.random.rand(nOutput, n2)

    layer0 = np.zeros((n0, 1))
    layer1 = np.zeros((n1, 1))
    layer2 = np.zeros((n2, 1))
    output = np.zeros((nOutput, 1))

    error = np.zeros((nOutput, 1))
     
    for iteration in xrange(iterations):
        
        sumerror = 0
        for sample in xrange(len(X)):
            nnOutput = forwardPass([layer0, layer1, layer2, output], [synapse1, synapse2, synapse3], X[sample])
            nsynapse1, nsynapse2, nsynapse3 = synapse1, synapse2, synapse3
            error = y[sample] - nnOutput[3]
            sumerror = sumerror + error
            
            layer0, layer1, layer2, layer3 = nnOutput            
            
            for j in xrange(n2):
                for i in xrange(nOutput):
                    nsynapse3[i][j] += rate*error[i]*sigmoid(nnOutput[3][i], True)*layer2[j]
            
            for k in xrange(n1):
                for j in xrange(n2):
                    for i in xrange(nOutput):
                        nsynapse2[j][k] += rate*error[i]*sigmoid(nnOutput[3][i], True)*synapse3[i][j]*sigmoid(nnOutput[2][j], True)*layer1[k]
             
            for l in xrange(n0):
                for k in xrange(n1):
                    for j in xrange(n2):
                        for i in xrange(nOutput):
                            nsynapse1[k][l] += rate*error[i]*sigmoid(nnOutput[3][i], True)*synapse3[i][j]*sigmoid(nnOutput[2][j], True)*synapse2[j][k]*sigmoid(nnOutput[1][k], True)*layer0[l]
        
        synapse1, synapse2, synapse3 = nsynapse1, nsynapse2, nsynapse3
                        
        if iteration%1000 == 0:
            print 'Iterations:', iteration, ' Error:', sumerror
            plotx.append(iteration)
            ploty.append(sumerror)
        
    plt.plot(plotx, ploty)
    plt.show()


def forwardPass(layer, synapse, X):
    layer[0] = X
    for weights in xrange(len(synapse)):
        if weights == 0:
            layer[1] = sigmoid(np.dot(synapse[0], layer[0]))
        else:
            layer[weights + 1] = sigmoid(np.dot(synapse[weights], layer[weights]))
    return layer
            



X = np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
y = np.array([ [0,1,1,0,1,0,0,1] ]).T

train(X, y)
