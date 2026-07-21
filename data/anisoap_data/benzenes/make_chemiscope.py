import numpy as np
from ase.io import read
from chemiscope import show
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA
from skmatter.decomposition import PCovR
import pickle

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

atom_frames = read("benzenes.xyz", ":")

lr2 = pickle.load(open("models/lr_baseline.sav", "rb"))
y_rem = np.load("representations/y_baselined.npy")
yr_scaler = pickle.load(open("models/yr_scaler.sav", 'rb'))

x = np.load("representations/x.npy")
y = np.load("representations/y.npy")
energies = np.load("representations/energies.npy")
xa = np.load("representations/xa.npy")
ta = np.load("representations/ta.npy")
tae = np.load("representations/tae.npy")

y_scaler = pickle.load(open("models/y_scaler.sav", "rb"))

# Just Some Visualization Things

# The dataset according to AniSOAP

pca = PCA(n_components=3)
t_pca = pca.fit_transform(x)
plt.scatter(t_pca[:, 0], t_pca[:, 1], c=energies)
plt.show()

# The dataset according to SOAP


pca = PCA(n_components=3)
ta_pca = pca.fit_transform(xa)
plt.scatter(ta_pca[:, 0], ta_pca[:, 1], c=energies)
plt.show()

# Visualizing which chemistries we do poorly on
pcovr = PCovR(n_components=3, mixing=0.5, regressor=lr2)
ta_pcovr = pcovr.fit_transform(xa, y_rem)
plt.scatter(ta_pcovr[:, 0], ta_pcovr[:, 1], c=energies)
plt.show()

properties_dict = {}
properties_dict["T (PCovR, energy errors)"] = ta_pcovr
properties_dict["T (PCA, AniSOAP)"] = t_pca
properties_dict["T (PCA, SOAP)"] = ta_pca
properties_dict["energy"] = energies
properties_dict["predicted_energy"] = energies.flatten() - y_rem.flatten()
properties_dict["energy_errors"] = y_rem
widget = show(atom_frames, properties=properties_dict)
widget.save("../../figures/si/benzenes.json")
