import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math  # Import math module for factorial

# System settings
hbar = 1.0  # Reduced Planck's constant
mass = 1.0  # Mass of the particle
omega = 1.0  # Angular frequency
n_levels = 10  # Number of energy levels
x_max = 4.0  # Maximum x-range
N = 500  # Number of points in the x-grid
x = np.linspace(-x_max, x_max, N)  # Spatial grid

# Energy levels for the harmonic oscillator
energy_levels = [hbar * omega * (n + 0.5) for n in range(n_levels)]

# Hermite polynomial calculation
def hermite_polynomial(n, x):
    """Calculate Hermite polynomials."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        h_prev = np.ones_like(x)
        h_curr = 2 * x
        for k in range(2, n + 1):
            h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
            h_prev, h_curr = h_curr, h_next
        return h_curr

# Wavefunction calculation
def wavefunction(n, x):
    """Calculate the wavefunction ψ_n(x)."""
    norm_factor = 1.0 / np.sqrt(2**n * math.factorial(n)) * (mass * omega / (np.pi * hbar))**0.25
    return norm_factor * np.exp(-mass * omega * x**2 / (2 * hbar)) * hermite_polynomial(n, np.sqrt(mass * omega / hbar) * x)

# Prepare the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-0.5, energy_levels[-1] + 1)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("Energy / |ψ(x)|²", fontsize=12)
ax.set_title("Quantum Harmonic Oscillator", fontsize=14)

# Plot potential and energy levels
potential = 0.5 * mass * omega**2 * x**2
ax.plot(x, potential, color='red', label="V(x) = 0.5 m ω² x²")
lines = [] 
for n in range(n_levels):
    energy = energy_levels[n]
    psi_n = wavefunction(n, x)
    lines.append(ax.plot(x, psi_n**2 + energy, label=f"E{n}")[0])

ax.legend()

# Animation function to update wavefunctions
def update(frame):
    for n, line in enumerate(lines):
        psi_n = wavefunction(n, x)
        phase = np.cos(2 * np.pi * frame / 100)  # Adding time-dependent oscillation
        line.set_ydata((psi_n**2) * np.abs(phase) + energy_levels[n])
    return lines

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
