#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:27:20 2024

@author: clairehamilton
"""

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import eval_legendre  # Placeholder for a wave function basis
from scipy.integrate import quad

def svd_factorization(wave_function):
    """
    Step 1: SVD factorization of target wave function, returns lambdas and alphas in eq. 70
    """
    U, s, Vh = svd(wave_function, full_matrices=False)
    lambda1, lambda2 = s[:2] #lambda = singular value of normalized two-photon wave function
    # V = -1*Vh.transpose()
    alpha1 = (U[:,0])
    alpha2 = (U[:,1])
    return lambda1, lambda2, alpha1, alpha2

def compute_a(lambda1, lambda2, alpha1, alpha2):
    """
    Step 2: calculate a^{(1,2)}_n = C^{(1,2)}[\sqrt{λ^1}α^1_n ± i\sqrt{λ^2}α^2_n].
    """
    C1 = np.sqrt(2) #?
    C2 = np.sqrt(2) #?
    a1 = C1*(np.sqrt(lambda1)*alpha1+1j*np.sqrt(lambda2)*alpha2)
    a2 = C2*(np.sqrt(lambda1)*alpha1-1j*np.sqrt(lambda2)*alpha2)
    return a1, a2

def compute_modulation_frequency(aX_n, t, Omega):
    """
    Calculate \omega(t) = Im(d/dt ln(\sum_n a_n e^{-i n \Omega t})).
    """
    n = np.arange(-size // 2, size // 2) 
    inner_sum = np.sum(aX_n[:, np.newaxis] * np.exp(-1j * n[:, np.newaxis] * Omega * t), axis=0) #inner sum
    derivative = -1j*Omega * np.sum(n[:, np.newaxis] * aX_n[:, np.newaxis] * np.exp(-1j * n[:, np.newaxis] * Omega * t), axis=0) #derivative
    omega_t = 1j*derivative/inner_sum 
    omega_t_real = np.real(omega_t)
    omega_t_imag = np.imag(omega_t)
    return omega_t_real, omega_t_imag

def fidelity(target_wavefunction, modulated_wavefunction): 
    """
    Compute fidelity F = |sum_{n1, n2} Psi*_{n1, n2} psi_{n1, n2}|^2.
    """
    return np.abs(np.sum(target_wavefunction.conjugate()*modulated_wavefunction))**2
    
def compute_a_from_modulated_frequency(omega, Omega, t, size):
    
    n = np.arange(-size // 2, size // 2)
    a = []
    for n_i in n:
        coefficient = np.sum(omega * np.exp(1j * n_i * Omega * t)) / len(t)
        a.append(coefficient)
    return np.array(a)

def construct_psi(size, a1, a2):
    psi = np.zeros((size, size), dtype=complex)
    for n1 in range(size):
        for n2 in range(size):
            psi[n1, n2] = a1[n1] * a2[n2] + a2[n1] * a1[n2]
    return 2*np.sqrt(2)*np.sqrt(2)*psi


def modulation_frequency_bell_state(Omega, t):
    omega1 = (np.sqrt(2)*Omega*np.sin(Omega*t))/(1+1j*np.sqrt(2)*np.cos(Omega*t))
    omega1_real = np.real(omega1)
    omega1_imag = np.imag(omega1)
    omega2 = -(np.sqrt(2)*Omega*np.sin(Omega*t))/(1-1j*np.sqrt(2)*np.cos(Omega*t))
    omega2_real = np.real(omega2)
    omega2_imag = np.imag(omega2)
    return omega1_real, omega1_imag, omega2_real, omega2_imag

def normalize_wavefunction(wavefunction):
    norm = np.sqrt(np.sum(np.abs(wavefunction)**2))
    if norm == 0:
        raise ValueError("cannot be normalized")
    normalized_wavefunction = wavefunction / norm
    
    return normalized_wavefunction

#%%
size = 5  # Size of frequency indices
Psi = np.zeros((size, size), dtype=complex)
# Bell state example: \Psi_{n1, n2} = (f_{n1} f_{n2} + g_{n1} g_{n2}) / sqrt(2)
f_n = np.zeros(size)
g_n = np.zeros(size)
f_n[2] = 1
g_n[1] = g_n[3] = 1 / np.sqrt(2)
for n1 in range(size):
    for n2 in range(size):
        Psi[n1, n2] = (f_n[n1] * f_n[n2] + g_n[n1] * g_n[n2]) / np.sqrt(2)

target_wavefunction = Psi #making target function bell state

# Step 1: SVD factorization
lambda1, lambda2, alpha1, alpha2 = svd_factorization(Psi)
print(f"Singular values: {lambda1}, {lambda2}")
print(f"Alpha1: {alpha1}")
print(f"Alpha2: {alpha2}")

# Step 2: Compute annihalation operators
a = compute_a(lambda1, lambda2, alpha1, alpha2)
a1 = a[0]
a2 = a[1]
print(f"a_n^{(1)}: {a1}")
print(f"a_n^{(2)}: {a2}")

Omega = 1.0  # Modulation frequency
t = np.linspace(0, 2*np.pi, 100)

omega1_real, omega1_imag = compute_modulation_frequency(a1, t, Omega)
omega2_real, omega2_imag = compute_modulation_frequency(a2, t, Omega)

omega1_actual_real, omega1_actual_imag, omega2_actual_real, omega2_actual_imag = modulation_frequency_bell_state(Omega, t)
#%%
plt.figure(dpi=600)
plt.plot(t, omega1_actual_real, label=r'actual $\omega^{(1)}(t)$')
plt.plot(t, omega2_actual_real, label=r'actual $\omega^{(2)}(t)$')
plt.plot(t, omega1_real, label=r'calculated $\omega^{(1)}(t)$')
plt.plot(t, omega2_real, label=r'calculated $\omega^{(2)}(t)$')
plt.xlabel("Time (t)")
plt.ylabel(r"Re{$\omega(t)$}")
plt.legend()
plt.title(f"Modulation Frequencies for $\Omega$ = {Omega}")
plt.show()
#%%
plt.figure(dpi=600)
plt.plot(t, omega1_actual_imag, label=r'actual $\omega^{(1)}(t)$')
plt.plot(t, omega2_actual_imag, label=r'actual $\omega^{(2)}(t)$')
plt.plot(t, omega1_imag, label=r'calculated $\omega^{(1)}(t)$')
plt.plot(t, omega2_imag, label=r'calculated $\omega^{(2)}(t)$')
plt.xlabel("Time (t)")
plt.ylabel(r"Im{$\omega(t)$}")
plt.legend()
plt.title(f"Modulation Frequencies for $\Omega$ = {Omega}")
plt.show()
#%%
calculated_a1 = compute_a_from_modulated_frequency(omega1_real, Omega, t, size)
calculated_a2 = compute_a_from_modulated_frequency(omega2_real, Omega, t, size)
modulated_waveform = construct_psi(size, calculated_a1, calculated_a2)

normalized_target_wavefunction = normalize_wavefunction(target_wavefunction) 
normalized_modulated_wavefunction = normalize_wavefunction(modulated_waveform) 

f = fidelity(normalized_target_wavefunction, normalized_modulated_wavefunction)
print(f)
#%%
# Define an objective function to minimize negative fidelity
def objective_function(Omega):
    t = np.linspace(0, 2 * np.pi, 100)
    omega1_real, omega1_imag = compute_modulation_frequency(a1, t, Omega)
    omega2_real, omega2_imag = compute_modulation_frequency(a2, t, Omega)

    calculated_a1 = compute_a_from_modulated_frequency(omega1_real, Omega, t, size)
    calculated_a2 = compute_a_from_modulated_frequency(omega2_real, Omega, t, size)
    modulated_wavefunction = construct_psi(size, calculated_a1, calculated_a2)

    normalized_modulated_wavefunction = normalize_wavefunction(modulated_wavefunction)
    normalized_target_wavefunction = normalize_wavefunction(target_wavefunction)

    f = fidelity(normalized_target_wavefunction, normalized_modulated_wavefunction)
    return -f  # Negate fidelity for minimization

# Optimization
result = minimize(objective_function, x0=1.0, bounds=[(0.1, 10)])  # Initial guess and bounds for Omega
optimized_Omega = result.x[0]
print(f"Optimized Omega: {optimized_Omega}")

# Recalculate modulated wavefunction with optimized Omega
t = np.linspace(0, 2 * np.pi, 100)
omega1_real, omega1_imag = compute_modulation_frequency(a1, t, optimized_Omega)
omega2_real, omega2_imag = compute_modulation_frequency(a2, t, optimized_Omega)

calculated_a1 = compute_a_from_modulated_frequency(omega1_real, optimized_Omega, t, size)
calculated_a2 = compute_a_from_modulated_frequency(omega2_real, optimized_Omega, t, size)
optimized_modulated_wavefunction = construct_psi(size, calculated_a1, calculated_a2)

# Normalized wavefunctions
normalized_optimized_modulated_wavefunction = normalize_wavefunction(optimized_modulated_wavefunction)
optimized_fidelity = fidelity(normalized_target_wavefunction, normalized_optimized_modulated_wavefunction)
print(f"Fidelity after optimization: {optimized_fidelity}")

# Heatmaps
def plot_heatmap(target, modulated):
    plt.figure(dpi=600)
    diff = np.abs(target - modulated)
    fig, ax = plt.subplots()
    c = ax.imshow(diff, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
    plt.colorbar(c, ax=ax)
    # ax.set_title(title)
    ax.set_xlabel("$n_1$")
    ax.set_ylabel("$n_2$")
    plt.show()

# Heatmap before optimization
plot_heatmap(normalized_target_wavefunction, normalized_modulated_wavefunction)

# Heatmap after optimization
plot_heatmap(normalized_target_wavefunction, normalized_optimized_modulated_wavefunction)

#%%
from scipy.optimize import minimize

# Define basis functions for \omega(t)
def basis_functions(t, num_terms=5):
    """Generate basis functions for \omega(t)."""
    basis = [np.ones_like(t)]  # Constant term
    for n in range(1, num_terms + 1):
        basis.append(np.sin(n * t))
        basis.append(np.cos(n * t))
    return np.array(basis)

def parameterized_omega(coefficients, t):
    """Construct \omega(t) from coefficients and basis functions."""
    basis = basis_functions(t, num_terms=(len(coefficients) - 1) // 2)
    return np.dot(coefficients, basis)

def objective_function(coefficients):
    """Objective function to maximize fidelity by optimizing \omega(t)."""
    t = np.linspace(0, 2 * np.pi, 100)
    omega = parameterized_omega(coefficients, t)

    # Compute a_n from \omega(t)
    calculated_a1 = compute_a_from_modulated_frequency(omega, Omega, t, size)
    calculated_a2 = compute_a_from_modulated_frequency(omega, Omega, t, size)

    # Construct the modulated wavefunction
    modulated_wavefunction = construct_psi(size, calculated_a1, calculated_a2)

    # Normalize wavefunctions
    normalized_modulated_wavefunction = normalize_wavefunction(modulated_wavefunction)
    normalized_target_wavefunction = normalize_wavefunction(target_wavefunction)

    # Compute fidelity
    return -fidelity(normalized_target_wavefunction, normalized_modulated_wavefunction)  # Negative for minimization

# Initial coefficients for \omega(t)
initial_coefficients = np.zeros(11)  # Adjust the number of terms as needed

# Optimize \omega(t)
result = minimize(objective_function, initial_coefficients, bounds=[(-10, 10)] * len(initial_coefficients))
optimized_coefficients = result.x
print(f"Optimized coefficients for \omega(t): {optimized_coefficients}")

# Generate optimized \omega(t)
t = np.linspace(0, 2 * np.pi, 100)
optimized_omega = parameterized_omega(optimized_coefficients, t)

# Compute the wavefunction with optimized \omega(t)
calculated_a1 = compute_a_from_modulated_frequency(optimized_omega, Omega, t, size)
calculated_a2 = compute_a_from_modulated_frequency(optimized_omega, Omega, t, size)
optimized_modulated_wavefunction = construct_psi(size, calculated_a1, calculated_a2)

# Normalize and calculate fidelity
normalized_optimized_modulated_wavefunction = normalize_wavefunction(optimized_modulated_wavefunction)
optimized_fidelity = fidelity(normalized_target_wavefunction, normalized_optimized_modulated_wavefunction)
print(f"Fidelity after optimizing \omega(t): {optimized_fidelity}")

# Heatmap
def plot_heatmap(target, modulated):
    diff = np.abs(target - modulated)
    fig, ax = plt.subplots()
    c = ax.imshow(diff, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
    plt.colorbar(c, ax=ax)
    ax.set_title("Hi")
    ax.set_xlabel("n1")
    ax.set_ylabel("n2")
    plt.show()

# Heatmap before optimization
plot_heatmap(normalized_target_wavefunction, normalized_modulated_wavefunction, "Heatmap Before Optimization")

# Heatmap after optimization
plot_heatmap(normalized_target_wavefunction, normalized_optimized_modulated_wavefunction, "Heatmap After Optimization")
#%%
plt.figure(dpi=600)
diff = np.abs(normalized_target_wavefunction - normalized_modulated_wavefunction)
fig, ax = plt.subplots()
c = ax.imshow(diff, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
plt.colorbar(c, ax=ax)
ax.set_title("|$\Psi_{n_1, n_2}-\psi_{n_1, n_2}|$ Before $\omega$ Optimization")
ax.set_xlabel("n1")
ax.set_ylabel("n2")
plt.show()
#%%
plt.figure(dpi=600)
diff = np.abs(normalized_target_wavefunction - normalized_optimized_modulated_wavefunction)
fig, ax = plt.subplots()
c = ax.imshow(diff, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
plt.colorbar(c, ax=ax)
ax.set_title("|$\Psi_{n_1, n_2}-\psi_{n_1, n_2}|$ After $\omega$ Optimization")
ax.set_xlabel("n1")
ax.set_ylabel("n2")
plt.show()