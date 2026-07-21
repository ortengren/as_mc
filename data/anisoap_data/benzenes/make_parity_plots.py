import matplotlib as mpl
import numpy as np
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from skmatter.preprocessing import StandardFlexibleScaler
import pickle

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})


def rmse(x, y):
    return np.sqrt(mse(x.flatten(), y.flatten()))

lr_args = dict(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=True)
lr = RidgeCV(**lr_args)

i_train = np.load("models/i_train.npy")
i_test = np.load("models/i_test.npy")

x = np.load("representations/x.npy")
y = np.load("representations/y.npy")
energies = np.load("representations/energies.npy")
xa = np.load("representations/xa.npy")
ta = np.load("representations/ta.npy")
tae = np.load("representations/tae.npy")

y_scaler = pickle.load(open("models/y_scaler.sav", "rb"))

# Model Time

# Model Time

x_train, x_test = x[i_train], x[i_test]
y_train, y_test = y[i_train], y[i_test]
xa_train, xa_test = xa[i_train], xa[i_test]
ta_train, ta_test = ta[i_train], ta[i_test]
tae_train, tae_test = tae[i_train], tae[i_test]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def plot(
    model,
    my_x,
    c=None,
    vmin=None,
    vmax=None,
    fig=None,
    ax=None,
    ax2=None,
    cmap="Blues",
    savename=None,
    show=True,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
    else:
        show = False

    if c is None:
        c = 1000 * np.array(
            y_scaler.inverse_transform(y)
            - y_scaler.inverse_transform(model.predict(my_x))
        )
    if vmax is None:
        vmax = c.max()
    if vmin is None:
        vmin = c.max()
    ax.plot(
        [-1000, 1000],
        [-1000, 1000],
        "r--",
        zorder=-2,
    )

    kwargs = dict(vmin=vmin, vmax=vmax, s=10, cmap=cmap)
    p = ax.scatter(
        y_scaler.inverse_transform(y_train),
        y_scaler.inverse_transform(model.predict(my_x[i_train])),
        c=c[i_train],
        marker="o",
        linewidth=0.3,
        label="train",
        rasterized=True,
        **kwargs,
    )
    ax.scatter(
        y_scaler.inverse_transform(y_test),
        y_scaler.inverse_transform(lr.predict(my_x[i_test])),
        c=c[i_test],
        marker="s",
        edgecolor="k",
        linewidth=0.5,
        label="test",
        rasterized=True,
        **kwargs,
    )

    if show:
        plt.colorbar(p, ax=ax, label="Error [meV/atom]")

    ax.set_xlabel("Energy [eV/atom]")
    ax.set_ylabel("Predicted Energy [eV/atom]")
    ax.set_aspect("equal")

    ax.annotate(
        xy=(0.05, 0.975),
        text=r"scores (train, test)"
        + "\n"
        + r"$R^2 = $({},{}){}$RMSE= $({},{})".format(
            round(model.score(my_x[i_train], y_train), 2),
            round(model.score(my_x[i_test], y_test), 2),
            str("\n"),
            round(
                1000
                * rmse(
                    y_scaler.inverse_transform(y_train),
                    y_scaler.inverse_transform(model.predict(my_x[i_train])),
                ),
                1,
            ),
            round(
                1000
                * rmse(
                    y_scaler.inverse_transform(y_test),
                    y_scaler.inverse_transform(model.predict(my_x[i_test])),
                ),
                1,
            ),
        ),
        xycoords="axes fraction",
        ha="left",
        va="top",
        bbox=dict(facecolor="aliceblue", edgecolor="grey", boxstyle="round"),
    )
    if ax2 is not None:
        ax2.hist(c[i_train], bins=100, histtype="step", label="train", color="grey")
        ax2.hist(c[i_test], bins=100, histtype="step", label="test", color="black")
        ax2.set_xlim([vmin, vmax])

        ax2.set_ylabel(
            "Histogram\nof Errors\n[meV/atom]",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )

        ax2.set_yscale("log")
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()


fig, axes = plt.subplots(
    3,
    3,
    figsize=(12, 6),
    gridspec_kw=dict(width_ratios=(1, 1, 1), height_ratios=(1, 0.1, 0.3)),
    sharey="row",
    sharex="row",
)

vmin = -134
vmax = 271
cmin = -vmin / (2 * vmax)

cmap = truncate_colormap(mpl.cm.seismic, cmin, 1.0)
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    label="Prediction error [meV/atom]",
    cax=axes[1][1],
    orientation="horizontal",
)
axes[1][0].axis("off")
axes[1][2].axis("off")

lr.fit(ta_train, y_train)
plot(lr, ta, vmin=vmin, vmax=vmax, fig=fig, ax=axes[0][0], ax2=axes[2][0], cmap=cmap)
axes[0][0].legend(loc="lower right")
axes[2][0].legend(loc="lower right")
axes[0][0].set_title("SOAP on atom centers")

lr.fit(x_train, y_train)
plot(lr, x, vmin=vmin, vmax=vmax, fig=fig, ax=axes[0][2], ax2=axes[2][2], cmap=cmap)
axes[2][2].set_ylabel("")
axes[0][2].set_ylabel("")
axes[0][2].set_title("AniSOAP")

lr.fit(tae_train, y_train)
plot(lr, tae, vmin=vmin, vmax=vmax, fig=fig, ax=axes[0][1], ax2=axes[2][1], cmap=cmap)
axes[2][1].set_ylabel("")
axes[0][1].set_ylabel("")
axes[0][1].set_title("SOAP on molecule centers")

axes[0][0].set_xlim([-133.48, -132.95])
axes[0][0].set_ylim([-133.52, -132.95])
fig.subplots_adjust(wspace=0.0, hspace=0.5)

plt.savefig("../../figures/benzenes-parity.pdf")
plt.show()

# A Model Built on AniSOAP with the remaining error learned by SOAP

lr.fit(x_train, y_train)

y_rem = y_scaler.inverse_transform(y) - y_scaler.inverse_transform(lr.predict(x))
yr_scaler = StandardFlexibleScaler(column_wise=True).fit(y_rem[i_train])

yr_train = yr_scaler.transform(y_rem[i_train])
yr_test = yr_scaler.transform(y_rem[i_test])

lr2 = RidgeCV(**lr_args)
lr2.fit(xa_train, yr_train)

yp = y_scaler.inverse_transform(lr.predict(x)) + yr_scaler.inverse_transform(
    lr2.predict(xa)
)
pickle.dump(lr2, open("models/lr_baseline.sav", "wb"))
pickle.dump(yr_scaler, open("models/yr_scaler.sav", "wb"))
np.save("representations/y_baselined.npy", y_rem)


fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

c = 1000 * np.array(energies.flatten() - yp.flatten())

ax.plot(
    [-1000, 1000],
    [-1000, 1000],
    "r--",
    zorder=-2,
)

kwargs = dict(vmin=vmin, vmax=vmax, s=10, cmap=cmap)
p = ax.scatter(
    y_scaler.inverse_transform(y_train),
    yp[i_train],
    c=c[i_train],
    marker="o",
    linewidth=0.3,
    label="train",
    rasterized=True,
    **kwargs,
)
ax.scatter(
    y_scaler.inverse_transform(y_test),
    yp[i_test],
    c=c[i_test],
    marker="s",
    edgecolor="k",
    linewidth=0.5,
    label="test",
    rasterized=True,
    **kwargs,
)

plt.colorbar(p, ax=ax, label="Error [meV/atom]")

ax.set_xlabel("Energy [eV/atom]")
ax.set_ylabel("Predicted Energy [eV/atom]")
ax.set_aspect("equal")

ax.set_xlim([-133.48, -132.95])
ax.set_ylim([-133.52, -132.95])
r2_train = r2_score(y_scaler.inverse_transform(y_train), yp[i_train])
r2_test = r2_score(y_scaler.inverse_transform(y_test), yp[i_test])
ax.annotate(
    xy=(0.05, 0.975),
    text=r"scores (train, test)"
    + "\n"
    + r"$R^2 = $({},{}){}$RMSE= $({},{})".format(
        round(r2_train, 2),
        round(r2_test, 2),
        str("\n"),
        round(
            1000
            * rmse(
                y_scaler.inverse_transform(y_train),
                yp[i_train],
            ),
            1,
        ),
        round(
            1000
            * rmse(
                y_scaler.inverse_transform(y_test),
                yp[i_test],
            ),
            1,
        ),
    ),
    xycoords="axes fraction",
    ha="left",
    va="top",
    bbox=dict(facecolor="aliceblue", edgecolor="grey", boxstyle="round"),
)

left, bottom, width, height = [0.475, 0.15, 0.25, 0.125]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.hist(c[i_train], bins=100, histtype="step", label="train", color="grey")
ax2.hist(c[i_test], bins=100, histtype="step", label="test", color="black")
ax2.set_xlim([vmin, vmax])

ax2.set_ylabel(
    "Histogram\nof Errors\n[meV/atom]", rotation=0, ha="right", va="center", labelpad=8
)

ax2.set_yscale("log")
plt.savefig("../../figures/benzenes-parity-corrected.pdf")
plt.show()
