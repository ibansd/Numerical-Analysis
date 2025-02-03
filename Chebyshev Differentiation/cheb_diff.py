import numpy as np
import matplotlib.pyplot as plt

def cheb_diff(f, n):
    """
    Compute Chebyshev differentiation matrix and derivative approximation.
    
    Parameters:
    f : callable - Function to differentiate
    n : int - Number of intervals (matrix will be (n+1)×(n+1))
    
    Returns:
    fp : ndarray - Derivative values at Chebyshev points
    D : ndarray - Chebyshev differentiation matrix
    """
    # Compute Chebyshev points
    xs = -np.cos(np.linspace(0, np.pi, n+1))
    
    # Evaluate function at Chebyshev points
    fs = f(xs)
    
    # Initialize differentiation matrix
    D = np.zeros((n+1, n+1))
    
    # Compute extradiagonal terms
    for j in range(n+1):  # Python 0-based indexing for j in [0,n]
        # Set d_j coefficient (j+1 to match MATLAB 1-based indexing check)
        d_j = 2 if (j == 0 or j == n) else 1
        
        for k in range(n+1):  # Python 0-based indexing for k in [0,n]
            # Set d_k coefficient (k+1 to match MATLAB 1-based indexing check)
            d_k = 2 if (k == 0 or k == n) else 1
            
            if j != k:
                # Note: xs[j] already correctly indexes because xs was created with Python indexing
                D[j, k] = (d_j/d_k) * ((-1)**(j+k)) / (xs[j] - xs[k])
    
    # Compute diagonal terms
    for j in range(n+1):
        D[j, j] = -np.sum(D[j, :])
    
    # Compute derivative approximation (matrix-vector product)
    # Note: fs needs to be reshaped as column vector for matrix multiplication
    fp = D @ fs
    
    return fp, D

# Test code
if __name__ == "__main__":
    n = 5
    xs = -np.cos(np.linspace(0, np.pi, n+1))
    
    # Test function and its derivative
    f = lambda x: np.sinh(x)
    fp = lambda x: np.cosh(x)
    
    # Get Chebyshev approximation
    fp_cheb, D = cheb_diff(f, n)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(xs, fp_cheb, 'b+', label='cheb')
    plt.plot(xs, fp(xs), 'r-', label='true')
    
    plt.grid(True)
    plt.xlim([-1.3, 1.2])
    plt.ylim([1, 1.8])
    plt.legend(loc='upper right')
    
    plt.show()
    
    # Print the matrix to compare with MATLAB reference
    print("Differentiation matrix D:")
    np.set_printoptions(precision=4, suppress=True)
    print(D)