import numpy as np
import pandas as pd
import random
from scipy.spatial.transform import Rotation

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


def get_rand_energy(frame, old_energy, rand=None):
    if rand is None:
        rand = random.Random()
    e_dif = 0.05 * rand.uniform(-1, 1)
    return old_energy + e_dif


def gay_berne_uniaxial(rij, ui_hat, uj_hat, kappa=3, kappa_prime=5, mu=2, nu=1):
    """
    This is the uniaxial gb used by Berardi and Zannoni in their Liquid crystals paper:
    https://pubs.rsc.org/en/content/articlelanding/1993/FT/FT9938904069
    """
    eps0 = 1
    sigma_s = 1
    sigma_e = kappa * sigma_s
    rij_mag = np.linalg.norm(rij)
    rij_hat = rij/rij_mag
    chi = (kappa**2 - 1) / (kappa**2 + 1)
    chi_prime = (kappa_prime**(1/mu) - 1) / (kappa_prime**(1/mu) + 1)
    common = (np.dot(ui_hat, rij_hat) + np.dot(uj_hat, rij_hat))**2 / (1 + chi_prime * np.dot(ui_hat, uj_hat)) + (np.dot(ui_hat, rij_hat) - np.dot(uj_hat, rij_hat))**2 / (1 - chi_prime * np.dot(ui_hat, uj_hat))
    eps_prime = 1 - 0.5 * chi_prime * common
    eps = (1 - chi**2 * np.dot(ui_hat, uj_hat)**2)**-0.5
    eps = eps0 * eps_prime**mu * eps ** nu
    sigma = sigma_s * (1 - 0.5 * chi * common) ** -0.5
    #TODO: Look into units to make sure they align with rest of algorithm
    return 4 * eps * ((sigma_s / (rij_mag - sigma + sigma_s))**12 - (sigma_s / (rij_mag - sigma + sigma_s))**6)


def gay_berne_walsh(
        frame,
        sigma_0=1,
        sigma_c=3.7496,
        eps_0=0.01036,
        eps_x=0.01036*5.7136,
        eps_y=0.01036*5.7136,
        eps_z=0.01036*0.0447,
        mu=2,
        nu=1,
):
    """
    Walsh, T. R. Towards an Anisotropic Bead-Spring Model for Polymers: A Gay-Berne Parametrization for Benzene.
    Molecular Physics 2002, 100 (17), 2867–2876. https://doi.org/10.1080/00268970210148796.

    :param frame:
    :param sigma_0: scaling parameter of ellipsoid axes, units of Å
    :param sigma_c: controls width of potential well
    :param eps_0: scaling parameter, units of eV/benzene (set to 1 kJ/mol in paper)
    :param eps_x: units of eV/benzene (units of kJ/mol in paper)
    :param eps_y: units of eV/benzene (units of kJ/mol in paper)
    :param eps_z: units of eV/benzene (units of kJ/mol in paper)
    :param mu: dimensionless parameter
    :param nu: dimensionless parameter
    :return: potential of system, units of eV/benzene (in units of kJ/mol in paper)
    """
    r_12 = frame.get_distance(0, 1, mic=True, vector=True) # units of Å
    r_12_mag = np.linalg.norm(r_12)
    r_12_hat = r_12 / r_12_mag
    # set ellipsoid semiaxis lengths using first ellipsoid; we assume second is the same
    sigma_x, sigma_y, sigma_z = sigma_0 * [5.8311, 5.8311, 4.9465] # values from paper, units of Å
    S = np.diag([sigma_x, sigma_y, sigma_z])
    # define orientational quantities
    R_1 = Rotation.from_quat(np.roll(frame.arrays["c_q"][0], -1))
    M_1 = R_1.as_matrix()
    R_2 = Rotation.from_quat(np.roll(frame.arrays["c_q"][1], -1))
    M_2 = R_2.as_matrix()
    # calculate shape parameter sigma
    A = (M_1.T @ S @ S @ M_1) + (M_2.T @ S @ S @ M_2)
    sigma = (2 * r_12_hat.T @ np.linalg.inv(A) @ r_12_hat)**(-1/2)
    # 1 kJ/mol ~ 0.01036 eV/benzene
    conv_factor = 0.01036
    # calculate strength parameter epsilon
    eps = (sigma_x * sigma_y + sigma_z**2) * ((2 * sigma_x * sigma_y / np.linalg.det(A))**(-1/2))
    E = np.diag([(eps_x/eps_0)**(1/mu), (eps_y/eps_0)**(1/mu), (eps_z/eps_0)**(1/mu)])
    B = (M_1.T @ E @ M_1) + (M_2.T @ E @ M_2)
    eps_prime = 2 * r_12_hat.T @ np.linalg.inv(B) @ r_12_hat
    epsilon = eps**nu * eps_prime**mu
    # calculate Gay-Berne potential
    t_1 = (sigma_c / (r_12_mag - sigma + sigma_c))**12
    t_2 = (sigma_c / (r_12_mag - sigma + sigma_c))**6
    U_GB = 4 * eps_0 * epsilon * (t_1 - t_2)
    return U_GB


if __name__ == "__main__":
    props = read_props("Pentacene/Pentacene/traj-pentacene.prop")
    print(props)

