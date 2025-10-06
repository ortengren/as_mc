import numpy as np
import pandas as pd
import random

def read_props(filename):
    props = {
        "lattice_energy": [],
        "heuristic_class": [],
        "cluster_cutoff_3a": [],
        "cluster_cutoff_5a": [],
    }
    with open(filename) as f:
        _ = f.readline()
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
    common = (np.dot(ui_hat, rij_hat) + np.dot(uj_hat, rij_hat))**2 / (1 + chi_prime * np.dot(ui_hat, uj_hat))    +    (np.dot(ui_hat, rij_hat) - np.dot(uj_hat, rij_hat))**2 / (1 - chi_prime * np.dot(ui_hat, uj_hat))
    eps_prime = 1 - 0.5 * chi_prime * common
    eps = (1 - chi**2 * np.dot(ui_hat, uj_hat)**2)**-0.5
    eps = eps0 * eps_prime**mu * eps ** nu
    sigma = sigma_s * (1 - 0.5 * chi * common) ** -0.5
    return 4 * eps * ((sigma_s / (rij_mag - sigma + sigma_s))**12 - (sigma_s / (rij_mag - sigma + sigma_s))**6)


if __name__ == "__main__":
    props = read_props("Pentacene/Pentacene/traj-pentacene.prop")
    print(props)
