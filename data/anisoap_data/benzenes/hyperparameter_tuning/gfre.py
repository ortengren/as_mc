#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import metatensor
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
from ase.io import read
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.target_space import NotUniqueError
from rascaline import SoapPowerSpectrum
from skmatter.metrics import global_reconstruction_error as GRE
from skmatter.preprocessing import StandardFlexibleScaler
from tqdm.auto import tqdm


def generate_anisoaps(frames, l_max, n_max, cutoff_radius, gaussian, mycg, a1, a3):
    for frame in frames:
        frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))
    representation = EllipsoidalDensityProjection(
        max_angular=l_max,
        max_radial=n_max,
        radial_basis_name="gto",
        rotation_type="quaternion",
        rotation_key="quaternions",
        cutoff_radius=cutoff_radius,
        radial_gaussian_width=gaussian,
        basis_rcond=1e-8,
        basis_tol=1e-4,
    )
    
    rep_raw = representation.transform(frames, show_progress=True)
    aniso_nu1 = standardize_keys(rep_raw)
    aniso_nu2 = cg_combine(
        aniso_nu1,
        aniso_nu1,
        clebsch_gordan=mycg,
        lcut=0,
        other_keys_match=["types_center"],
    )
    rep = metatensor.operations.mean_over_samples(aniso_nu2, sample_names="center")
    return rep.block().values.squeeze()


def go(e10, e12, gaussian, verbose=True):
    e10 = round(e10, 2)
    e12 = round(e12, 2)
    g = round(gaussian, 2)
    key = make_key(e10, e12, g)
    if key in dump_dict:
        gre = float(dump_dict[key])
    else:
        x = generate_anisoaps(
            frames,
            l_max,
            n_max,
            cutoff_radius,
            gaussian,
            mycg,
            a1=e10,
            a3=e12,
        )
        gre = GRE(x, xa)
        dump_dict[key] = gre
        np.savez("saved_gfres.npz", **dump_dict)
    if verbose:
        print(key, gre)
    return -gre


def read_key(key):
    a1, a3, g = key.split("-")
    a1 = float(a1[2:])
    a3 = float(a3[2:])
    g = float(g[1:])
    return a1, a3, g


def make_key(e10, e12, g):
    return "A1{}-A3{}-G{}".format(round(e10, 3), round(e12, 3), round(g, 3))


if __name__ == "__main__":
    l_max = 9
    n_max = 6
    cutoff_radius = 7.0
    mycg = ClebschGordanReal(l_max)
    gaussians = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    HYPER_PARAMETERS = {
        "cutoff": cutoff_radius,
        "max_radial": n_max,
        "max_angular": l_max,
        "atomic_gaussian_width": 0.5,
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.01},
        },
    }

    if not os.path.exists("selected.npy"):
        all_atom_frames = read("../benzenes.xyz", ":")

        calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
        descriptor = calculator.compute(all_atom_frames)
        descriptor = descriptor.keys_to_samples("center_type")
        descriptor = descriptor.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )
        descriptor = metatensor.operations.mean_over_samples(
            descriptor, sample_names=["center_type", "atom"]
        )

        Xa_raw = descriptor.block().values.squeeze()
        xa_scaler = StandardFlexibleScaler(column_wise=False).fit(Xa_raw)
        xa = xa_scaler.transform(Xa_raw)

        rs = np.random.RandomState(0)
        selected = rs.choice(np.arange(len(xa)), 1000)
        np.save("selected.npy", selected)
    else:
        selected = np.load("selected.npy")

    frames = read("../ellipsoids.xyz", ":")
    atom_frames = read("../benzenes.xyz", ":").copy()
    frames = [frames[s] for s in selected]
    atom_frames = [atom_frames[s] for s in selected]

    for frame in frames:
        frame.pbc = True
        if "c_q" in frame.arrays and "quaternions" not in frame.arrays:
            frame.arrays["quaternions"] = frame.arrays["c_q"]

    calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
    descriptor = calculator.compute(atom_frames)
    descriptor = descriptor.keys_to_samples("center_type")
    descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    descriptor = metatensor.operations.mean_over_samples(
        descriptor, sample_names=["center_type", "atom"]
    )

    Xa_raw = descriptor.block().values.squeeze()
    xa_scaler = StandardFlexibleScaler(column_wise=False).fit(Xa_raw)
    xa = xa_scaler.transform(Xa_raw)

    if os.path.exists("saved_gfres.npz"):
        dump_dict = dict(np.load("saved_gfres.npz"))
    else:
        dump_dict = {}
        np.savez("saved_gfres.npz", **dump_dict)

    pbounds = {
        "e10": (0.5, 5.5001),
        "e12": (0.5, 5.5001),
        "gaussian": (0.5, 3.01),
    }

    targets = []
    optimizer = BayesianOptimization(
        f=go, pbounds=pbounds, random_state=1, verbose=2, allow_duplicate_points=True
    )

    priming_pairs = [
        {"e10": round(i, 3), "e12": round(j, 3), "gaussian": l}
        for j in np.arange(0.5, 5.51, 0.5)
        for i in np.arange(0.5, 5.51, 0.5)
        for l in gaussians
    ]

    for e10 in np.arange(*pbounds["e10"], 0.25):
        for g in gaussians:
            priming_pairs.append(
                {"e10": round(e10, 3), "e12": round(e10, 3), "gaussian": g}
            )

    stored_pairs = dict(dump_dict)
    priming_pairs = [
        p for p in priming_pairs if make_key(*p.values()) not in stored_pairs
    ]
    np.random.shuffle(priming_pairs)

    n_total = 100
    n_done = 0

    pbar = tqdm(total=len(priming_pairs))

    try:
        for pair_key in stored_pairs:
            pair = read_key(pair_key)
            try:
                optimizer.probe(
                    params={
                        "e10": pair[0],
                        "e12": pair[1],
                        "gaussian": pair[2],
                    },
                    lazy=False,
                )
                targets.append(optimizer.space.target.max())
                n_done += 1
            except NotUniqueError:
                pass

        for p in priming_pairs:
            # if make_key(*p.values()) not in stored_pairs:
            # _ = stored_pairs.pop(make_key(*p.values()))
            optimizer.probe(params=p, lazy=False)
            targets.append(optimizer.space.target.max())
            n_done += 1
            pbar.update(1)

        print(optimizer.res)

        print("{} left".format(max(100, n_total - n_done)))
        optimizer.maximize(
            init_points=5 if n_done < 5 else 0,
            n_iter=max(100, n_total - n_done),
        )
        targets.append(optimizer.space.target.max())

    except KeyboardInterrupt:
        pass

    if not os.path.exists("optimized_gfres.npz"):
        optimized_gfres = {}
        if input("Save?: {}\t".format(optimizer.max["params"])).upper() == "Y":
            optimized_gfres["optimized_semiaxes"] = (
                optimizer.max["params"]["e10"],
                optimizer.max["params"]["e10"],
                optimizer.max["params"]["e12"],
            )
            optimized_gfres["optimized_gaussian"] = optimizer.max["params"]["gaussian"]
            input(optimized_gfres)
            np.savez("optimized_gfres.npz", **optimized_gfres)
    optimized_gfres = np.load("optimized_gfres.npz")

    if "optimized_semiaxes" in optimized_gfres:
        pbounds["e10"] = [0.1, 5.51]
        pbounds["e12"] = [0.1, 5.51]
        print(
            optimized_gfres["optimized_semiaxes"], optimized_gfres["optimized_gaussian"]
        )
        pairs = []
        for e10 in np.arange(*pbounds["e10"], 0.1):
            pairs.append((round(e10, 2), round(e10, 2)))

        # for e10 in np.arange(round(optimized_gfres["optimized_semiaxes"][0]-0.5, 1),
        #                      round(optimized_gfres["optimized_semiaxes"][0]+0.5, 1), 0.1):
        for e10 in np.arange(*pbounds["e10"], 0.2):
            for e12 in np.arange(*pbounds["e12"], 0.2):
                pairs.append((round(e10, 2), round(e12, 2)))

        pairs = list(
            sorted(
                set(pairs),
                key=lambda x: (x[0], x[1]),
            )
        )
        input(pairs)
        for p in tqdm(pairs):
            try:
                go(
                    *p,
                    gaussian=float(optimized_gfres["optimized_gaussian"]),
                    verbose=True
                )
            except ValueError:
                print("FAIL", p)
