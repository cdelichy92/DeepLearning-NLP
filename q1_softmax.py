import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    """

    if len(x.shape) == 1:
        c = np.max(x)
        x = np.exp(x-c)
        x = x/np.sum(x)
        return x
    
    c = np.amax(x, axis=1).reshape(x.shape[0],1)
    x = np.exp(x-c)
    x = x/np.sum(x,axis=1,keepdims=True)
    
    return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

if __name__ == "__main__":
    test_softmax_basic()