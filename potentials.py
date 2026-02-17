import ase
import numpy as np
from numpy.linalg import norm
import pandas as pd
import random
from scipy.spatial.transform import Rotation

EPS_0 = 8.8541878188E-22 # F / Å

def read_props(filename):
    props = {
        "lattice_energy": [],
        "heuristic_class": [],
        "cluster_cutoff_3a": [],
        "cluster_cutoff_5a": [],
    }
    with open(filename) as f:
        # throw away the first line
        _ = f.readline()
        # iterate through lines of file and update dict
        l = f.readline()
        while l != "":
            l = l.split()
            if len(l) != 4:
                print(l)
            props["lattice_energy"].append(l[0])
            props["heuristic_class"].append(l[1])
            props["cluster_cutoff_3a"].append(l[2])
            props["cluster_cutoff_5a"].append(l[3])
            l = f.readline()
    return props


WALSH_PARAMS = {
    "sigma_0": 1.,      # Å
    "sigma_c": 3.7496,  # Å
    "sigma_x": 5.8311,  # Å
    "sigma_y": 5.8311,  # Å
    "sigma_z": 4.9465,  # Å
    "eps_0": 1.,        # kJ / mol
    "eps_x": 5.7136,    # kJ / mol
    "eps_y": 5.7136,    # kJ / mol
    "eps_z": 0.0447,    # kJ / mol
    "mu": 7.6093,
    "nu": -12.46,
    "Theta": 9.2074     # atomic units
}


def gb_shape_function(uhat1, uhat2, rhat, sigma0, kappa):
    chi = gb_shape_anisotropy(kappa)
    term1 = (np.dot(uhat1, rhat) + np.dot(uhat2, rhat))**2 / (1 + chi*np.dot(uhat1, uhat2))
    term2 = (np.dot(uhat1, rhat) - np.dot(uhat2, rhat))**2 / (1 - chi*np.dot(uhat1, uhat2))
    sigma = sigma0 / np.sqrt(1 - (chi / 2)*(term1 + term2))
    return sigma


def gb_shape_anisotropy(kappa):
    return (kappa**2 - 1)/(kappa**2 + 1)


def gb_axial_energy(uhat1, uhat2, kappa):
    chi = gb_shape_anisotropy(kappa)
    return 1 / np.sqrt(1 - (chi*np.dot(uhat1, uhat2))**2)


def gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu):

    chi_prime = gb_energy_anisotropy(kappa_prime, mu)
    term1 = (np.dot(uhat1, rhat) + np.dot(uhat2, rhat))**2 / (1 + chi_prime*np.dot(uhat1, uhat2))
    term2 = (np.dot(uhat1, rhat) - np.dot(uhat2, rhat))**2 / (1 - chi_prime*np.dot(uhat1, uhat2))
    return 1 - chi_prime*(term1 + term2) / 2


def gb_energy_anisotropy(kappa_prime, mu):
    return (kappa_prime**(1/mu) - 1) / (kappa_prime**(1/mu) + 1)


def gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu):
    eps1 = gb_axial_energy(uhat1, uhat2, kappa)
    eps2 = gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu)
    return eps0 * eps1**nu * eps2**mu


def gb(uhat1, uhat2, r, sigma0, eps0, kappa, kappa_prime, mu, nu):
    rhat = r / norm(r)
    eps = gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu)
    sigma = gb_shape_function(uhat1, uhat2, rhat, sigma0, kappa)
    term = sigma0 / (norm(r) - sigma + sigma0)
    return 4 * eps * (term**12 - term**6)


def quadrupole(uhat1, uhat2, r, Theta):
    return NotImplementedError


GB_PARAMS = {
    "sigma0": 5.6908734316048575,
    "eps0": 0.5274639566548358,
    "kappa": 0.5105882064300075,
    "kappa_prime": 0.7730283946074973,
    "mu": 2.,
    "nu": 1.,
}


# --- Helper Functions for Vectorization ---

# Replaces np.dot(a, b) for arrays of shape (N, 3)
def vdot(v1, v2):
    return np.sum(v1 * v2, axis=1)

# Replaces norm(v) for arrays of shape (N, 3)
def vnorm(v):
    return norm(v, axis=1)


# --- Vectorized Physics Functions ---

def vec_gb_shape_function(uhat1, uhat2, rhat, sigma0, kappa):
    chi = gb_shape_anisotropy(kappa)
    # Use vdot instead of np.dot
    dot_u1_r = vdot(uhat1, rhat)
    dot_u2_r = vdot(uhat2, rhat)
    dot_u1_u2 = vdot(uhat1, uhat2)
    term1 = (dot_u1_r + dot_u2_r) ** 2 / (1 + chi * dot_u1_u2)
    term2 = (dot_u1_r - dot_u2_r) ** 2 / (1 - chi * dot_u1_u2)
    sigma = sigma0 / np.sqrt(1 - (chi / 2) * (term1 + term2))
    return sigma


def vec_gb_axial_energy(uhat1, uhat2, kappa):
    chi = gb_shape_anisotropy(kappa)
    # Use vdot
    return 1 / np.sqrt(1 - (chi * vdot(uhat1, uhat2)) ** 2)


def vec_gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu):
    chi_prime = gb_energy_anisotropy(kappa_prime, mu)
    # Use vdot
    dot_u1_r = vdot(uhat1, rhat)
    dot_u2_r = vdot(uhat2, rhat)
    dot_u1_u2 = vdot(uhat1, uhat2)

    term1 = (dot_u1_r + dot_u2_r) ** 2 / (1 + chi_prime * dot_u1_u2)
    term2 = (dot_u1_r - dot_u2_r) ** 2 / (1 - chi_prime * dot_u1_u2)
    return 1 - chi_prime * (term1 + term2) / 2


def vec_gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu):
    eps1 = vec_gb_axial_energy(uhat1, uhat2, kappa)
    eps2 = vec_gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu)
    return eps0 * eps1 ** nu * eps2 ** mu


def vec_gb(uhat1, uhat2, r, sigma0, eps0, kappa, kappa_prime, mu, nu):
    # Vectorized normalization
    # We must reshape the norm to (N, 1) so we can divide the (N, 3) vector r
    r_magnitudes = vnorm(r)
    rhat = r / r_magnitudes[:, np.newaxis]

    eps = vec_gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu)
    sigma = vec_gb_shape_function(uhat1, uhat2, rhat, sigma0, kappa)

    # Calculate final term
    term = sigma0 / (r_magnitudes - sigma + sigma0)
    return 4 * eps * (term ** 12 - term ** 6)