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


def pairwise_gay_berne_walsh(
        frame,
        idx_1,
        idx_2,
        sigma_0,
        sigma_c,
        sigma_x,
        sigma_y,
        sigma_z,
        eps_0,
        eps_x,
        eps_y,
        eps_z,
        mu,
        nu,
):
    """
    Walsh, T. R. Towards an Anisotropic Bead-Spring Model for Polymers: A Gay-Berne Parametrization for Benzene.
    Molecular Physics 2002, 100 (17), 2867–2876. https://doi.org/10.1080/00268970210148796.
    Assumes frame has only two particles.

    :param frame:
    :param sigma_0: scaling parameter of ellipsoid axes, units of Å
    :param sigma_c: controls width of potential well
    :param eps_0: scaling parameter (set to 1 kJ/mol in paper)
    :param eps_x: (units of kJ/mol in paper)
    :param eps_y: (units of kJ/mol in paper)
    :param eps_z: (units of kJ/mol in paper)
    :param mu: dimensionless parameter
    :param nu: dimensionless parameter
    :return: potential of system (in units of kJ/mol in paper)
    """
    r_12 = frame.get_distance(idx_1, idx_2, mic=True, vector=True) # units of Å
    r_12_mag = np.linalg.norm(r_12)
    r_12_hat = r_12 / r_12_mag
    # set ellipsoid semiaxis lengths using first ellipsoid; we assume second is the same
    sigma_x, sigma_y, sigma_z = sigma_0 * [sigma_x, sigma_y, sigma_z]
    S = np.diag([sigma_x, sigma_y, sigma_z])
    # define orientational quantities
    R_1 = Rotation.from_quat(np.roll(frame.arrays["c_q"][idx_1], -1))
    M_1 = R_1.as_matrix()
    R_2 = Rotation.from_quat(np.roll(frame.arrays["c_q"][idx_2], -1))
    M_2 = R_2.as_matrix()
    # calculate shape parameter sigma
    A = (M_1.T @ S @ S @ M_1) + (M_2.T @ S @ S @ M_2)
    sigma = (2 * np.dot(r_12_hat, np.linalg.inv(A) @ r_12_hat))**(-1/2)
    # calculate strength parameter epsilon
    E = eps_0**(1/mu) * np.diag(((1/eps_x)**(1/mu), (1/eps_y)**(1/mu), (1/eps_z)**(1/mu)))
    B = (M_1.T @ E @ M_1) + (M_2.T @ E @ M_2)
    nu_term = ((sigma_x*sigma_y + sigma_z**2) * np.sqrt(np.linalg.det(A)) / np.sqrt(2*sigma_x*sigma_y))**nu
    mu_term = (2 * np.dot(r_12_hat, np.linalg.inv(B) @ r_12_hat))**mu
    eps = nu_term * mu_term
    # calculate Gay-Berne potential
    t_1 = (sigma_c / (r_12_mag - sigma + sigma_c))**12
    t_2 = (sigma_c / (r_12_mag - sigma + sigma_c))**6
    U_GB = 4 * eps_0 * eps * (t_1 - t_2)
    return U_GB


def pairwise_quadrupole_potential(frame, idx_1, idx_2, Theta):
    r_12 = frame.get_distance(idx_1, idx_2, mic=True, vector=True)  # units of Å
    r_12_mag = np.linalg.norm(r_12)
    r_12_hat = r_12 / r_12_mag
    n_1 = frame.arrays["or_vec"][idx_1]
    n_1_hat = n_1 / np.linalg.norm(n_1)
    n_2 = frame.arrays["or_vec"][idx_2]
    n_2_hat = n_2 / np.linalg.norm(n_2)
    factor = 0.75 * (35*(np.dot(n_1_hat, r_12_hat)**2)*(np.dot(n_2_hat, r_12_hat)**2)
                     - 5*np.dot(n_1_hat, r_12_hat)**2 - 5*np.dot(n_2_hat, r_12_hat)**2 - 20*np.dot(n_1_hat, r_12_hat)**2
                     * np.dot(n_2_hat, r_12_hat)*np.dot(n_1_hat, n_2_hat) + 2*np.dot(n_1_hat, n_2_hat)**2 + 1)
    U_QQ = factor * Theta**2 / (4*np.pi*EPS_0 * r_12_mag**5)
    return U_QQ


def gay_berne_walsh(
        frame,
        idx,
        sigma_0,
        sigma_c,
        sigma_x,
        sigma_y,
        sigma_z,
        eps_0,
        eps_x,
        eps_y,
        eps_z,
        mu,
        nu,
):
    U_GB = 0
    for i, _ in enumerate(frame):
        if i == idx:
            continue
        U_GB += pairwise_gay_berne_walsh(
            frame, idx, i, sigma_0, sigma_c, sigma_x, sigma_y, sigma_z, eps_0, eps_x, eps_y, eps_z, mu, nu)
    return U_GB


def quadrupole_potential(
        frame,
        idx_1,
        Theta,
):
    U_QQ = 0
    for i, _ in enumerate(frame):
        if i == idx_1:
            continue
        U_QQ += pairwise_quadrupole_potential(frame, idx_1, i, Theta)
    return U_QQ


def calc_walsh_potential(
        frame,
        idx,
        sigma_0,
        sigma_c,
        sigma_x,
        sigma_y,
        sigma_z,
        eps_0,
        eps_x,
        eps_y,
        eps_z,
        mu,
        nu,
        Theta,
):
    U_GB = 0
    U_QQ = 0
    for i, _ in enumerate(frame):
        if i == idx:
            continue
        U_GB += pairwise_gay_berne_walsh(
            frame, idx, i, sigma_0, sigma_c, sigma_x, sigma_y, sigma_z, eps_0, eps_x, eps_y, eps_z, mu, nu)
        U_QQ += pairwise_quadrupole_potential(frame, idx, i, Theta)
    return U_QQ + U_GB


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