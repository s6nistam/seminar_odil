import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x, t):
    """Compute the exact analytical solution at given x and t."""
    total = 0.0
    for i in range(1, 6):
        total += np.cos((x + 0.5) * i * np.pi) * np.cos(t * i * np.pi)
        # total += np.cos(x * i * np.pi) * np.cos((t - 0.5) * i * np.pi)
    return total / 5.0

def main():
    # Parameters
    Nx = 256     # Number of spatial points (including boundaries)
    Nt = Nx     # Number of time steps (to satisfy CFL condition)
    L = 2.0       # Spatial domain length (from -1 to 1)
    T = 1.0       # Final time
    
    # Grid setup
    dx = L / (Nx - 1)
    dt = T / Nt
    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, T, Nt + 1)
    
    # Stability parameter (must be <= 1 for stability)
    r = (dt / dx) ** 2
    print(f"Grid: dx = {dx:.4f}, dt = {dt:.4f}, r = {r:.4f} (CFL condition: r <= 1)")

    # Initialize solution array
    u = np.zeros((Nt + 1, Nx))
    
    # Initial condition (t = 0)
    u[0, :] = exact_solution(x, 0)
    
    # First time step using initial velocity condition (u_t = 0)
    # Apply boundary conditions for i=1
    u[1, 0] = exact_solution(x[0], t[1])
    u[1, -1] = exact_solution(x[-1], t[1])
    
    # Interior points for first time step
    for i in range(1, Nx - 1):
        u[1, i] = u[0, i] + 0.5 * r * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])
    
    # Time stepping (i >= 1)
    for i in range(1, Nt):
        # Apply Dirichlet boundary conditions
        u[i + 1, 0] = exact_solution(x[0], t[i + 1])
        u[i + 1, -1] = exact_solution(x[-1], t[i + 1])
        
        # Update interior points using finite difference scheme
        for j in range(1, Nx - 1):
            u[i + 1, j] = 2 * u[i, j] - u[i - 1, j] + r * (
                u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]
            )
    
    # Compute exact solution at final time
    exact_final = exact_solution(x, T)
    
    # Calculate L2 error
    error = np.sqrt(dx * np.sum((u[Nt, :] - exact_final) ** 2))
    print(f"L2 error at t=1: {error:.6e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, u[Nt, :], 'b-', linewidth=2, label='Numerical Solution')
    plt.plot(x, exact_final, 'r--', linewidth=2, label='Exact Solution')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x, 1)', fontsize=12)
    plt.title('1D Wave Equation Solution at t=1', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()