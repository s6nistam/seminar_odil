import numpy as np
import matplotlib.pyplot as plt

def u_exact(x, t):
    """Analytical solution u(x, t)"""
    s = np.zeros_like(x)
    for i in range(1, 6):
        s += np.cos(x * i * np.pi) * np.cos(i * np.pi * (t - 0.5))
    return s / 5.0

def u_t_exact(x, t):
    """Analytical time derivative ∂u/∂t"""
    s = np.zeros_like(x)
    for i in range(1, 6):
        s += np.cos(x * i * np.pi) * (-i * np.pi * np.sin(i * np.pi * (t - 0.5)))
    return s / 5.0

def main():
    # Domain parameters
    x_min, x_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0
    L = x_max - x_min  # Spatial domain length
    
    # Discretization parameters
    N_x = 256  # Number of spatial points (must be odd for symmetry)
    dx = L / (N_x - 1)
    cfl = 0.5  # CFL number (must be < 1 for stability)
    dt = cfl * dx  # Time step satisfying CFL condition
    N_t = int((t_max - t_min) / dt) + 1  # Number of time steps
    
    # Create grids
    x = np.linspace(x_min, x_max, N_x)
    t = np.linspace(t_min, t_max, N_t)
    
    # Initialize solution arrays
    U = np.zeros((N_t, N_x))  # Numerical solution u(x,t)
    
    # Set initial conditions using exact solution
    U[0, :] = u_exact(x, t[0])
    
    # Compute first time step using exact initial derivative
    # Boundary conditions at t=0 (already set in U[0,:])
    U[1, 0] = u_exact(x[0], t[1])  # Left boundary
    U[1, -1] = u_exact(x[-1], t[1])  # Right boundary
    
    # Interior points for n=1 using Taylor expansion
    for i in range(1, N_x - 1):
        # u(x, Δt) ≈ u(x,0) + Δt*u_t(x,0) + (Δt²/2)*u_xx(x,0)
        U[1, i] = U[0, i] + dt * u_t_exact(x[i], t[0]) + \
                 (dt**2 / (2 * dx**2)) * (U[0, i+1] - 2*U[0, i] + U[0, i-1])
    
    # Time stepping (leapfrog scheme)
    for n in range(1, N_t - 1):
        # Apply Dirichlet boundary conditions
        U[n+1, 0] = u_exact(x[0], t[n+1])
        U[n+1, -1] = u_exact(x[-1], t[n+1])
        
        # Update interior points using finite differences
        for i in range(1, N_x - 1):
            U[n+1, i] = 2*U[n, i] - U[n-1, i] + \
                        (dt**2 / dx**2) * (U[n, i+1] - 2*U[n, i] + U[n, i-1])
    
    # Compute exact solution for error analysis
    U_exact_all = np.zeros_like(U)
    for n in range(N_t):
        U_exact_all[n, :] = u_exact(x, t[n])
    
    # Calculate errors
    u_error = np.abs(U - U_exact_all)
    max_u_error = np.max(u_error)
    
    print(f"Maximum solution error: {max_u_error:.4e}")
    print(f"Grid: {N_x} spatial points, {N_t} time steps")
    print(f"dx = {dx:.6f}, dt = {dt:.6f}, CFL = {dt/dx:.4f}")
    
    # Create 2D heatmap plot
    plt.figure(figsize=(12, 8))
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(x, t)
    
    # Plot numerical solution
    plt.pcolormesh(X, T, U, shading='auto', cmap='viridis', 
                   vmin=np.min(U_exact_all), vmax=np.max(U_exact_all))
    plt.colorbar(label='u(x,t)')
    
    # Add contour lines for better visualization
    contour = plt.contour(X, T, U, colors='white', alpha=0.3, linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title(f'1D Wave Equation Solution (Error: {max_u_error:.2e})\n'
              f'Grid: {N_x}×{N_t}, CFL: {cfl}', fontsize=14)
    
    # Add analytical solution boundary markers
    plt.plot(x, np.ones_like(x)*t_min, 'w-', linewidth=2, label='t=0 (initial)')
    plt.plot(x, np.ones_like(x)*t_max, 'w--', linewidth=2, label='t=1 (final)')
    plt.plot(np.ones_like(t)*x_min, t, 'w:', linewidth=2, label='x=-1 (boundary)')
    plt.plot(np.ones_like(t)*x_max, t, 'w-.', linewidth=2, label='x=1 (boundary)')
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('wave_2d_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()