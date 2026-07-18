# -*- coding: utf-8 -*-
"""
Naressi Law Spin Chain Simulation Framework
Author: Fabian Leo Naressi
Description: Comprehensive simulation suite for analyzing scaling properties and 
             Von Neumann Entropy in 1D spin chains with Golden Ratio geometric coupling.
             Includes Dense QuTiP, Sparse SciPy, TeNPy MPS implementations, and 
             formal mathematical verification suites.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib
# Configura il backend headless PRIMA di importare pyplot
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

# --- CONFIGURAZIONE DEFINITIVA POST-BUG ---
matplotlib.rcParams['text.usetex'] = False          # Disabilita il compilatore esterno
matplotlib.rcParams['font.family'] = 'sans-serif'    # RISOLVE RASTER OVERFLOW: Cambia da serif a sans-serif
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans' # RISOLVE RASTER OVERFLOW: Usa DejaVu per la matematica
matplotlib.rcParams['figure.max_open_warning'] = 0    # Evita warning sulle figure aperte
matplotlib.rcParams['font.size'] = 10               # Riduce il carico dei vettori
# Try importing QuTiP and TeNPy for completeness, but handle exceptions if not installed
try:
    import qutip as qt
except ImportError:
    qt = None

try:
    from tenpy.networks.mps import MPS
    from tenpy.models.spins import SpinChain
    from tenpy.algorithms import dmrg
except ImportError:
    MPS, SpinChain, dmrg = None, None, None

def get_naressi_coupling(N, alpha, phi=(1 + np.sqrt(5)) / 2):
    """
    Computes the geometric coupling constants according to the Naressi Law:
    J_n = alpha * phi^-n * exp(-n / phi^2)
    """
    n = np.arange(N - 1)
    return alpha * (phi**(-n)) * np.exp(-n / (phi**2))

# =============================================================================
# 1. DENSE QUTIP IMPLEMENTATION (Optimal for N <= 16)
# =============================================================================
def build_hamiltonian_qutip(N, alpha, g, phi=(1 + np.sqrt(5)) / 2):
    if qt is None:
        raise ImportError("QuTiP is required for this function.")
    J = get_naressi_coupling(N, alpha, phi)
    si = qt.qeye(2)
    sx = qt.sigmax()
    sz = qt.sigmaz()
    
    H = 0
    # Ising interaction term
    for n in range(N - 1):
        op_list = [si] * N
        op_list[n] = sx
        op_list[n+1] = sx
        H -= J[n] * qt.tensor(op_list)
        
    # Transverse field term
    for n in range(N):
        op_list = [si] * N
        op_list[n] = sz
        H -= g * qt.tensor(op_list)
        
    return H

def calculate_vn_entropy_qutip(N, alpha, g, phi=(1 + np.sqrt(5)) / 2):
    if qt is None:
        raise ImportError("QuTiP is required for this function.")
    
    # 1. Calcoliamo l'autovettore fondamentale usando SciPy Sparse (veloce e robusto)
    H_sparse = build_hamiltonian_sparse(N, alpha, g, phi)
    _, evec_sparse = eigsh(H_sparse, k=1, which='SA')
    
    # 2. Convertiamo l'autovettore in un Qobj di QuTiP (Ket dello stato fondamentale)
    dims = [[2]*N, [1]*N]
    gs = qt.Qobj(evec_sparse[:, 0], dims=dims)
    
    # 3. Calcolo dell'entropia di Von Neumann a metà catena
    rho_red = gs.ptrace(range(N // 2))
    return qt.entropy_vn(rho_red)
# =============================================================================
# 2. SPARSE SCIPY IMPLEMENTATION (Optimal for 16 < N <= 24)
# =============================================================================
def build_hamiltonian_sparse(N, alpha, g, phi=(1 + np.sqrt(5)) / 2):
    """
    Efficiently constructs the Transverse-Field Ising model Hamiltonian 
    using Compressed Sparse Row (CSR) matrices.
    """
    J = get_naressi_coupling(N, alpha, phi)
    I = sp.eye(2, format='csr')
    X = sp.csr_matrix([[0, 1], [1, 0]])
    Z = sp.csr_matrix([[1, 0], [0, -1]])
    
    H = sp.csr_matrix((2**N, 2**N))
    
    # Ising Interaction: Sum J_n * X_n * X_{n+1}
    for n in range(N - 1):
        op = 1
        for i in range(N):
            if i == n or i == n + 1:
                op = sp.kron(op, X, format='csr')
            else:
                op = sp.kron(op, I, format='csr')
        H -= J[n] * op
        
    # Transverse Field: Sum g * Z_n
    for n in range(N):
        op = 1
        for i in range(N):
            if i == n:
                op = sp.kron(op, Z, format='csr')
            else:
                op = sp.kron(op, I, format='csr')
        H -= g * op
        
    return H

def run_sparse_eigenanalysis(N, alpha, g, phi=(1 + np.sqrt(5)) / 2):
    H = build_hamiltonian_sparse(N, alpha, g, phi)
    evals, evecs = eigsh(H, k=1, which='SA') 
    return evals[0], evecs[:, 0]

# =============================================================================
# 3. MATRIX PRODUCT STATES / DMRG IMPLEMENTATION (Optimal for N >= 50)
# =============================================================================
def run_tenpy_dmrg(N=100, alpha=1.618, g=1.0, chi_max=64, phi=(1 + np.sqrt(5)) / 2):
    if dmrg is None:
        print("TeNPy is not installed. Returning conceptual placeholder.")
        return None
    
    J_list = list(get_naressi_coupling(N, alpha, phi))
    
    model_params = {
        'L': N,
        'S': 0.5,
        'J': J_list, 
        'g': g, 
        'bc_MPS': 'finite'
    }
    
    model = SpinChain(model_params)
    psi = MPS.from_product_state(model.lat.mps_sites(), [0]*N, bc=model.lat.bc_MPS)
    
    dmrg_params = {
        'mixer': True,
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10}
    }
    
    info = dmrg.run(model, psi, dmrg_params)
    entropy = psi.entanglement_entropy()
    
    return info['E'], entropy

# =============================================================================
# 4. SCIENTIFIC VERIFICATION SUITE (Scale-Invariance & Falsifiability)
# =============================================================================
def verify_scale_invariance(N_sizes=[4, 8, 12, 16], alpha=1.61803398875, g=1.0):
    """
    Executes a thermodynamic finite-size analysis to map the scaling behavior
    of the ground-state energies across progressive architectural layers.
    """
    print("\n" + "="*70)
    print(" VERIFICATION SUITE I: FINITE-SIZE SCALING & RATIO ANALYSIS")
    print("="*70)
    energies = {}
    
    for N in N_sizes:
        E0, _ = run_sparse_eigenanalysis(N, alpha, g)
        energies[N] = E0
        print(f"Lattice Size N={N:2d} | Ground State Energy E0: {E0:.6f}")
        
    print("\n--- Structural Scaling Ratios ---")
    sizes = list(energies.keys())
    for idx in range(1, len(sizes)):
        n_curr = sizes[idx]
        n_prev = sizes[idx-1]
        ratio = energies[n_curr] / energies[n_prev]
        print(f"Energy Ratio E0(N={n_curr}) / E0(N={n_prev}): {ratio:.6f}")

def verify_falsifiability_perturbation(N=12, alpha=1.61803398875, g=1.0, epsilon=0.05):
    """
    Tests system resilience against geometric configuration breakdown.
    Compares ideal golden coupling vs a perturbed geometry (phi + epsilon).
    """
    print("\n" + "="*70)
    print(" VERIFICATION SUITE II: GEOMETRIC BREAKDOWN & FALSIFIABILITY")
    print("="*70)
    if qt is None:
        print("QuTiP is required to evaluate reduced density matrix entropy. Skipping.")
        return
        
    phi_ideal = (1 + np.sqrt(5)) / 2
    phi_perturbed = phi_ideal + epsilon
    
    S_ideal = calculate_vn_entropy_qutip(N, alpha, g, phi=phi_ideal)
    S_perturbed = calculate_vn_entropy_qutip(N, alpha, g, phi=phi_perturbed)
    
    print(f"Lattice Setup (N={N}, alpha={alpha}, g={g})")
    print(f"-> Ideal System Von Neumann Entropy (phi = {phi_ideal:.5f}): {S_ideal:.6f}")
    print(f"-> Perturbed System Von Neumann Entropy (phi + epsilon = {phi_perturbed:.5f}): {S_perturbed:.6f}")
    print(f"-> Net Structural Entropy Shift (Delta S): {S_perturbed - S_ideal:.6f}")

# =============================================================================
# AUTOMATION AND VISUALIZATION LOOP (QuTiP Example)
# =============================================================================
def plot_scaling_analysis():
    if qt is None:
        print("QuTiP not found. Cannot run standard automated plot.")
        return
        
    N_values = [4, 8, 12, 16]
    g = 1.0
    alpha_range = np.linspace(0.5, 2.5, 20)
    results = {n: [] for n in N_values}

    print("\nStarting automated scaling plot visualization...")
    for n in N_values:
        for a in alpha_range:
            entropy = calculate_vn_entropy_qutip(n, a, g)
            results[n].append(float(entropy))

    # Completely reset and clean the figure canvas
    plt.close('all')
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in N_values:
        ax.plot(alpha_range, results[n], label=f'N={n}')
        
    # --- SANITIZED LABELS (Bypasses mathtext LaTeX parsing entirely) ---
    ax.set_xlabel('alpha (Scale Parameter)') 
    ax.set_ylabel('Von Neumann Entropy')
    ax.set_title('Naressi Law: Scaling Analysis')
    
    ax.legend()
    ax.grid(True)
    
    # Save using standard layout adjustments
    fig.savefig('naressi_scaling_plot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved successfully as 'naressi_scaling_plot.png'.")

if __name__ == '__main__':
    print("Naressi Law Framework Initialized successfully.")
    
    # Run the added predictive and formal check routines
    verify_scale_invariance(N_sizes=[4, 8, 12, 16])
    verify_falsifiability_perturbation(N=12, epsilon=0.05)
    
    # Optional plot execution
    plot_scaling_analysis()
