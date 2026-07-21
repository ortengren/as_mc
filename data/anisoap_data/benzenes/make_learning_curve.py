import numpy as np
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error as mse
import pickle
from tqdm.auto import tqdm

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

atom_frames = read("benzenes.xyz", ":")


def rmse(*x):
    return np.sqrt(mse(*x))


i_train = np.load("models/i_train.npy")
i_test = np.load("models/i_test.npy")

x = np.load("representations/x.npy")
y = np.load("representations/y.npy")
energies = np.load("representations/energies.npy")

x_train, x_test = x[i_train], x[i_test]
y_train, y_test = y[i_train], y[i_test]

y_scaler = pickle.load(open("models/y_scaler.sav", "rb"))

lr = RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=False)

cv = 20
train_sizes = np.arange(0.05, 1.01, 0.05)
n = []
ltr = []
lte = []

for t in tqdm(train_sizes, leave=False):
    my_n = int(t * len(x_train))
    train_errors = []
    test_errors = []
    for i in tqdm(range(cv), leave=False):
        my_i_train = np.random.choice(i_train, size=my_n)
        my_x = x[my_i_train]
        my_y = y[my_i_train]

        lr.fit(my_x, my_y)
        train_errors.append(
            rmse(energies[my_i_train], y_scaler.inverse_transform(lr.predict(my_x)))
        )
        test_errors.append(
            rmse(energies[i_test], y_scaler.inverse_transform(lr.predict(x_test)))
        )
    n.append(my_n)
    ltr.append(train_errors)
    lte.append(test_errors)

    print(train_errors)
    print(test_errors)

n = np.array(n)
ltr = np.multiply(1000, np.array(ltr))
lte = np.multiply(1000, np.array(lte))

b = plt.boxplot(positions=n, x=lte.T, vert=True, patch_artist=True, widths=200, labels=n)
plt.scatter(np.nan, np.nan, facecolor='r', edgecolor='k', marker='s', label='Test')

for patch in b['boxes']:
    patch.set_facecolor('r')
for patch in b['fliers']:
    patch.set_markeredgecolor('r')
for patch in b['medians']:
    patch.set_color('k')

b = plt.boxplot(positions=n, x=ltr.T, vert=True, patch_artist=True, widths=200, labels=n)
plt.scatter(np.nan, np.nan, facecolor='grey', edgecolor='k', marker='s', label='Train')

for patch in b['boxes']:
    patch.set_facecolor('grey')
for patch in b['fliers']:
    patch.set_markeredgecolor('k')
for patch in b['medians']:
    patch.set_color('k')
    
plt.legend()
plt.xlabel(r"$N_\text{train}$")
plt.ylabel("RMSE [meV/atom]")
plt.savefig("../../figures/si/learning_curve.pdf")
plt.show()
