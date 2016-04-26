import numpy as np
import random

def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x)
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        x_old_ix = x[ix]
        
        x[ix] = x_old_ix + h
        random.setstate(rndstate)
        forward_f = f(x)[0]
        
        x[ix] = x_old_ix - h
        random.setstate(rndstate)
        backward_f = f(x)[0]
        
        numgrad = (forward_f - backward_f)/(2*h)
        
        x[ix] = x_old_ix

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))    
    gradcheck_naive(quad, np.random.randn(3,))    
    gradcheck_naive(quad, np.random.randn(4,5))  
    print ""

if __name__ == "__main__":
    sanity_check()
    
