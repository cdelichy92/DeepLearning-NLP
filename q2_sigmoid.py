import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid function.
    """
    
    x = 1/(1+np.exp(-x))
    
    return x

def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. 
    """
    
    f = f*(1-f)
    
    return f

def test_sigmoid_basic():
    """
    Some simple tests to get you started. 
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    assert np.amax(f - np.array([[0.73105858, 0.88079708], 
        [0.26894142, 0.11920292]])) <= 1e-6
    print g
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])) <= 1e-6
    print "You should verify these results!\n"

if __name__ == "__main__":
    test_sigmoid_basic();
