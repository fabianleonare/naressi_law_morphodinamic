
# Quantum Morphodynamics & The Naressi Law

This repository contains the official computational framework and theoretical documentation for the **Naressi Law** and its applications within Quantum Morphodynamics, Quasicrystalline Architectures, and Multi-Stage Marx Generators.

## Overview
The **Naressi Law** establishes a paradigm of *Holistic Renormalization*, challenging the classical assumption of scale independence by introducing an explicit structural constraint tied to the Golden Ratio ($\gr$). 

$$J_n = \alpha \cdot \gr^{-n} \cdot e^{-\frac{n}{\gr^2}}$$

Rather than treating microscopic and macroscopic fields as decoupled systems, this framework maps the deterministic geometric flow across hierarchical steps, preventing ultraviolet singularities and defining automatic wave localization profiles.

## Repository Structure
* `/simulation`: Python suite utilizing sparse matrices (`scipy.sparse`) and exact diagonalization (`qutip`) to evaluate ground-state energy convergence ($E_0^{(n)}/E_0^{(n-1)} \to \gr$) and von Neumann entropy.
* `/paper`: Comprehensive LaTeX source files (`main.tex`, `references.bib`) pre-configured for academic journal submission (JCSA layout).

## Getting Started
### Prerequisites
Ensure you have Python 3.8+ installed, then clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
## Getting Started

### Prerequisites
Il framework è ottimizzato per ambienti isolati tramite Anaconda/Conda. Assicurati di avere Conda installato sul tuo sistema host.

### Installazione e Uso Locale

1. Clona la repository:
   ```bash
   git clone https://github.com/fabianleonare/naressi_law_morphodinamic.git
   cd naressi_law_morphodinamic
conda env create -f environment.yml
conda activate quantum
python naressi_simulation.py
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fabianleonare/naressi_law_morphodinamic/main)
