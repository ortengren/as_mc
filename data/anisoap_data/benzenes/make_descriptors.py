import metatensor
import numpy as np
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal, cg_combine, standardize_keys
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib import rc
from rascaline import SoapPowerSpectrum
from sklearn.decomposition import PCA
from skmatter.metrics import global_reconstruction_error as GRE
from sklearn.model_selection import train_test_split
from skmatter.preprocessing import StandardFlexibleScaler
import pickle

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})


l_max = 9
n_max = 6
mycg = ClebschGordanReal(l_max)

atom_frames = read("benzenes.xyz", ":")
frames = read("ellipsoids.xyz", ":")
energies = np.array([aframe.info["energy_pa"] for aframe in atom_frames])
plt.hist(energies, bins=100)
plt.xlabel("Loaded Energies, eV")
plt.show()


# Computing the AniSOAP Vectors

y_scaler = StandardFlexibleScaler(column_wise=False).fit(energies.reshape(-1, 1))
y = y_scaler.transform(energies.reshape(-1, 1))

(
    i_train,
    i_test,
) = train_test_split(np.arange(len(y)), test_size=0.1, shuffle=True)
np.save("models/i_train.npy", i_train)
np.save("models/i_test.npy", i_test)

rgw = float(np.load("hyperparameter_tuning/optimized_gfres.npz")["optimized_gaussian"])
a1 = float(
    np.load("hyperparameter_tuning/optimized_gfres.npz")["optimized_semiaxes"][0]
)
a2 = float(
    np.load("hyperparameter_tuning/optimized_gfres.npz")["optimized_semiaxes"][1]
)
a3 = float(
    np.load("hyperparameter_tuning/optimized_gfres.npz")["optimized_semiaxes"][2]
)
input((rgw, a1, a2, a3))
cutoff_radius = 7.0


for frame in frames:
    frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = a2 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))


representation = EllipsoidalDensityProjection(
    max_angular=l_max,
    max_radial=n_max,
    cutoff_radius=cutoff_radius,
    radial_basis_name="gto",
    radial_gaussian_width=rgw,
    subtract_center_contribution=True,
    rotation_key="c_q",
    basis_rcond=1e-8,
    basis_tol=1e-3,
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
x_raw = rep.block().values.squeeze()
x_scaler = StandardFlexibleScaler(column_wise=False).fit(x_raw[i_train])
x = x_scaler.transform(x_raw)
print(x.shape)


plt.title("Just a little plot to see that the vectors have some variance.")
plt.plot(x.T)
plt.twinx()
plt.semilogy(np.var(x, axis=0))
plt.show()


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
calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
descriptor = calculator.compute(atom_frames)
descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
descriptor = metatensor.operations.mean_over_samples(
    descriptor, sample_names=["center_type", "atom"]
)

Xa_raw = descriptor.block().values.squeeze()
xa_scaler = StandardFlexibleScaler(column_wise=False).fit(Xa_raw[i_train])

xa = xa_scaler.transform(Xa_raw)
ta = PCA(n_components=np.linalg.matrix_rank(x)).fit_transform(xa)
plt.plot(xa.T.mean(axis=0))
plt.show()
print(GRE(x_raw, Xa_raw))


HYPER_PARAMETERS = {
    "cutoff": cutoff_radius,
    "max_radial": n_max,
    "max_angular": l_max,
    "atomic_gaussian_width": a1,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.01},
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

descriptor = calculator.compute(frames)
descriptor = descriptor.keys_to_samples("center_type")
descriptor = descriptor.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
descriptor = metatensor.operations.mean_over_samples(
    descriptor, sample_names=["center_type", "atom"]
)

X_raw = descriptor.block().values.squeeze()
X_raw = X_raw[:, X_raw.var(axis=0) > 1e-12]

xae_scaler = StandardFlexibleScaler(column_wise=False).fit(X_raw[i_train])
xae = xae_scaler.transform(X_raw)
tae = PCA(
    n_components=min(np.linalg.matrix_rank(xae), np.linalg.matrix_rank(x))
).fit_transform(xae)

print(GRE(X_raw, Xa_raw))

np.save("representations/x_raw.npy", x_raw)
np.save("representations/x.npy", x)
np.save("representations/y.npy", y)
np.save("representations/energies.npy", energies)
np.save("representations/xa_raw.npy", Xa_raw)
np.save("representations/xa.npy", xa)
np.save("representations/xae.npy", xae)
np.save("representations/xae_raw.npy", X_raw)
np.save("representations/ta.npy", ta)
np.save("representations/tae.npy", tae)

pickle.dump(y_scaler, open("models/y_scaler.sav", "wb"))
pickle.dump(x_scaler, open("models/x_scaler.sav", "wb"))
pickle.dump(xa_scaler, open("models/xa_scaler.sav", "wb"))
pickle.dump(xae_scaler, open("models/xae_scaler.sav", "wb"))
