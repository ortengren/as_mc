import ase
import numpy as np
from ase.neighborlist import neighbor_list
from numpy import linalg as la
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
    chi = (kappa**2 - 1)/(kappa**2 + 1)
    term1 = (np.vecdot(uhat1, rhat) + np.vecdot(uhat2, rhat))**2 / (1 + chi*np.vecdot(uhat1, uhat2))
    term2 = (np.vecdot(uhat1, rhat) - np.vecdot(uhat2, rhat))**2 / (1 - chi*np.vecdot(uhat1, uhat2))
    sigma = sigma0 / np.sqrt(1 - (chi / 2)*(term1 + term2))
    return sigma


def gb_axial_energy(uhat1, uhat2, kappa):
    chi = (kappa**2 - 1)/(kappa**2 + 1)
    return 1 / np.sqrt(1 - (chi*np.vecdot(uhat1, uhat2))**2)


def gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu):
    chi_prime = (kappa_prime**(1/mu) - 1) / (kappa_prime**(1/mu) + 1)
    term1 = (np.vecdot(uhat1, rhat) + np.vecdot(uhat2, rhat))**2 / (1 + chi_prime*np.vecdot(uhat1, uhat2))
    term2 = (np.vecdot(uhat1, rhat) - np.vecdot(uhat2, rhat))**2 / (1 - chi_prime*np.vecdot(uhat1, uhat2))
    return 1 - chi_prime*(term1 + term2) / 2


def gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu):
    eps1 = gb_axial_energy(uhat1, uhat2, kappa)
    eps2 = gb_directional_energy(uhat1, uhat2, rhat, kappa_prime, mu)
    return eps0 * eps1**nu * eps2**mu


def gb(uhat1, uhat2, r, sigma0, eps0, kappa, kappa_prime, mu, nu):
    rmag = np.expand_dims(la.norm(r, axis=-1), axis=-1)
    rhat = r / rmag
    eps = gb_energy_function(uhat1, uhat2, rhat, eps0, kappa, kappa_prime, mu, nu)
    sigma = gb_shape_function(uhat1, uhat2, rhat, sigma0, kappa)
    term = sigma0 / (la.norm(r, axis=-1) - sigma + sigma0)
    return 4 * eps * (term**12 - term**6)


def quadrupole(uhat1, uhat2, r, Q):
    rmag = np.expand_dims(la.norm(r, axis=-1), axis=-1)
    rhat = r / rmag
    a1 = np.vecdot(uhat1, rhat)
    a2 = np.vecdot(uhat2, rhat)
    b12 = np.vecdot(uhat1, uhat2)
    prefactor = 0.75 * Q**2 / rmag**5
    prefactor = np.squeeze(prefactor)
    s = 1 + 2*b12**2 - 5*(a1**2 + a2**2) - 20*a1*a2*b12 + 35*(a1**2)*(a2**2)
    return prefactor * s


def get_total_energy(M, sigma0, eps0, kappa, kappa_prime, mu, nu, Q):
    # M should have shape (N, 1431, 3, 3) where N is the number of frames
    E_GB = gb(M[:, :, 0, :], M[:, :, 1, :], M[:, :, 2, :], sigma0, eps0, kappa, kappa_prime, mu, nu)
    E_QQ = quadrupole(M[:, :, 0, :], M[:, :, 1, :], M[:, :, 2, :], Q)
    E_QQ = np.squeeze(E_QQ)
    pw_energies = E_GB + E_QQ
    # pw_energies should have shape (N, 1431)
    energies = np.sum(pw_energies, axis=-1)
    return energies


def calc_total_energy(frame, nl_cutoff, method="GB"):
    if method == "GB":
        # get all interacting pairs (i, j) and their shift vectors
        i, j, s = neighbor_list("ijs", frame, nl_cutoff)

        # filter for i < j to avoid double counting
        unique_pairs_mask = i < j
        i = i[unique_pairs_mask]
        j = j[unique_pairs_mask]
        s = s[unique_pairs_mask]

        # calculate displacements
        cell = frame.get_cell()
        shift_vecs = np.dot(s, cell)
        displacements = frame.positions[j] + shift_vecs - frame.positions[i]

        # calculate orientations
        uhat1 = frame.arrays["or_vec"][i]
        uhat2 = frame.arrays["or_vec"][j]

        # calculate pairwise energies
        gb_e = gb(uhat1, uhat2, displacements, *GB_PARAMS.values())
        qq_e = np.squeeze(quadrupole(uhat1, uhat2, displacements, QQ))
        pw_energies = gb_e + qq_e

        return np.sum(pw_energies)
    else:
        return NotImplementedError()


GB_PARAMS = {
    "sigma0": 6.16753952,
    "eps0": 0.08,
    "kappa": 0.53134663,
    "kappa_prime": 0.68032599,
    "mu": -0.39313992,
    "nu": 4.37606907,
}

QQ = -3.83795985