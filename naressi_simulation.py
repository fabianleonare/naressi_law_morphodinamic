# -*- coding: utf-8 -*-
"""
Naressi Law Spin Chain Simulation Framework
Author: Fabian Leo Naressi
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use('Agg')  # Headless backend

import matplotlib.pyplot as plt

# ====================== CONFIGURAZIONE DEFINITIVA ======================
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'   # Sicuro per alpha
matplotlib.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.unicode_minus'] = False

# Try importing QuTiP and TeNPy...
try:
    import qutip as qt
except ImportError:
    qt = None

# ... (tutto il resto del codice rimane uguale fino a plot_scaling_analysis)

def plot_scaling_analysis():
    if qt is None:
        print("QuTiP not found. Cannot run standard automated plot.")
        return
        
    N_values = [4, 8, 12, 16]
    g = 1.0
    alpha_range = np.linspace(0.5, 2.5, 30)   # Più punti per curve più lisce
    results = {n: [] for n in N_values}
    
    print("\nStarting automated scaling plot visualization...")
    
    for n in N_values:
        for a in alpha_range:
            entropy = calculate_vn_entropy_qutip(n, a, g)
            results[n].append(float(entropy))
    
    plt.close('all')
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, n in enumerate(N_values):
        ax.plot(alpha_range, results[n], 
                linestyle=styles[i], 
                color=colors[i],
                linewidth=2.2,
                marker='o',
                markersize=4,
                label=f'N={n}')
    
    # === LABELS SICURI (NO LaTeX) ===
    ax.set_xlabel(r'$\alpha$ (Scale Parameter)', fontsize=12)   # r'' + $ $ è sicuro
    ax.set_ylabel('Von Neumann Entropy', fontsize=12)
    ax.set_title('Naressi Law: Scaling Analysis', fontsize=14, pad=15)
    
    ax.legend(# Salvataggio ad alta qualità
    fig.savefig('naressi_scaling_plot_2.png', dpi=300, bbox_inches='tight')
    
    # Versione alternativa (low-entropy zoom)
    ax.set_ylim(0, 0.4)
    fig.savefig('naressi_scaling_plot_1.png', dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    print("Plots salvati correttamente:")
    print("-> naressi_scaling_plot_2.png (range completo)")
    print("-> naressi_scaling_plot_1.png (low-entropy
