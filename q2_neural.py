import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    y_hat = softmax(z2)

    cost = -np.sum( labels * np.log(y_hat) )
    
    delta2 = y_hat - labels
    gradW2 = np.dot(h.T, delta2)
    gradb2 = np.sum(delta2, axis=0)
    
    delta1 = np.dot(delta2, W2.T)*sigmoid_grad(h)
    gradW1 = np.dot(data.T, delta1)
    gradb1 = np.sum(delta1, axis=0)
    
    ### Stack gradients
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,dimensions), params)

if __name__ == "__main__":
    sanity_check()