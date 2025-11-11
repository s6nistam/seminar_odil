import numpy as np

def compute_eoc(N, err):
    """
    Compute the Experimental Order of Convergence (EOC)
    
    Parameters:
    N (list or array): List of mesh sizes (e.g., number of grid points)
    err (list or array): Corresponding errors (e.g., L2 or max error)
    
    Returns:
    eoc (list): List of EOC values between consecutive mesh sizes
    """
    # Convert to numpy arrays
    N = np.array(N)
    err = np.array(err)
    
    # Check if we have at least 2 data points
    if len(N) < 2:
        raise ValueError("At least two data points are required to compute EOC.")
    
    # Check that N and err are monotonically increasing (refinement)
    if not np.all(np.diff(N) > 0):
        raise ValueError("N must be strictly increasing (mesh refinement).")
    
    if not np.all(np.diff(err) < 0):
        print("Warning: Errors are not strictly decreasing. EOC may not be meaningful.")
    
    # Compute EOC for each consecutive pair
    eoc = []
    for i in range(len(N) - 1):
        N_i = N[i]
        N_ip1 = N[i + 1]
        err_i = err[i]
        err_ip1 = err[i + 1]
        
        # Avoid division by zero or log of zero
        if err_ip1 == 0:
            raise ValueError(f"Error at N={N_ip1} is zero. Cannot compute EOC.")
        
        # Compute EOC
        eoc_i = np.log(err_i / err_ip1) / np.log(N_ip1 / N_i)
        eoc.append(eoc_i)
    
    return np.array(eoc)

# Example usage:
if __name__ == "__main__":
    # Example: N = [10, 20, 40, 80, 160] (mesh sizes)
    # err = [0.01, 0.0025, 0.000625, 0.00015625, 3.90625e-05] (errors)
    N = [16,32,64,128,256]
    err = [0.2707197656931557, 0.09139643936044689, 0.023548114046037655, 0.005744825080613608, 0.001421212783115855, 0.0003539060622771706]
    
    eoc_values = compute_eoc(N, err)
    
    print("Mesh Size (N) | Error | EOC")
    print("-" * 30)
    for i in range(len(N)):
        if i == 0:
            print(f"{N[i]:>10} | {err[i]:>7.2e} |")
        else:
            print(f"{N[i]:>10} | {err[i]:>7.2e} | {eoc_values[i-1]:>6.3f}")