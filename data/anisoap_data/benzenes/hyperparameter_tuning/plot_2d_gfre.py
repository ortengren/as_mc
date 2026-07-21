import os

import matplotlib as mpl
import matplotlib.tri as tri
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

from gfre import read_key

l_max = 9
n_max = 6
cutoff_radius = 7.0
dump_dict = dict(np.load("saved_gfres.npz"))

def plot_trisurf2d(tri, z, ax, vmax, vmin, *args, **kwargs):
    triangles = tri.get_masked_triangles()
    xt = tri.x[triangles]
    yt = tri.y[triangles]
    zt = z[triangles]
    verts = np.stack((xt, yt, zt), axis=-1)

    polyc = PolyCollection(verts[:, :, :2], *args, **kwargs)
    # average over the three points of each triangle
    avg_z = np.mean(verts[:, :, 2], axis=1)
    polyc.set_array(avg_z)
    polyc.set_clim(vmin, vmax)

    ax.add_collection(polyc)


vmin = 0.35
vmax = 0.7
raw_cmap = mpl.colormaps["jet"]

dl = 0.025
levels = np.arange(vmin, vmax + 0.01, dl)
colors = [raw_cmap((x - vmin) / (vmax - vmin)) for x in levels]
cmap = ListedColormap(colors, name="my_cmap")


# In[6]:


fig = plt.figure(figsize=(8, 6), layout="constrained")
ax_list = [
        ["c", "c", "c"],
        ["0.5", "1.0", "1.5"],
        ["2.0", "2.5", "3.0"],
    ]
ax_dict = fig.subplot_mosaic(
    ax_list,
    height_ratios=(0.1, 1, 1),
    width_ratios=(1,1,1),
    # sharex=True,
    # sharey=True,
)


for i, strg in enumerate(ax_dict.keys()):
    if strg !='c':
        g = float(strg)
        x = []
        y = []
        mgres = []

        for key in dump_dict:
            a1, a3, s = read_key(key)
            if np.isclose(s, g, 0.1) and a3>=0.1:
                x.append(a1)
                y.append(a3)
                mgres.append(float(dump_dict[key]))
        if len(x) > 3:
            ax = ax_dict[str(g)]
            print(
                g,
                min(mgres),
                x[np.argmin(mgres)],
                y[np.argmin(mgres)],
                max(mgres),
                x[np.argmax(mgres)],
                y[np.argmax(mgres)],

            )
            x = np.array(x)
            y = np.array(y)
            mgres = np.array(mgres)
            triang = tri.Triangulation(
                x,
                y,
            )

            ax.tricontour(
                triang,
                mgres,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zorder=-2,
                levels=levels,
            )
            p = ax.scatter(
                x,
                y,
                c=mgres,
                vmin=vmin - dl / 2.0,
                vmax=vmax + dl / 2.0,
                cmap=cmap,
                marker="o",
                s=6,  # facecolors='none',
                rasterized=True,
            )
            ax.scatter(
                x[np.argmin(mgres)],
                y[np.argmin(mgres)],
                c=min(mgres),
                vmin=vmin - dl / 2.0,
                vmax=vmax + dl / 2.0,
                cmap=cmap,
                marker="*",
                s=20,  # facecolors='none',
                rasterized=True,
            )

            ax.annotate(
                xy=(0.05, 0.95),
                text=r"$\sigma_{{GTO}}$ = {}".format(g),
                xycoords="axes fraction",
                ha="left",
                va="top",
                bbox=dict(facecolor='aliceblue', edgecolor='black', boxstyle='round')

            )
            ax.set_aspect("equal")
cb = plt.colorbar(
            p,
            cax=ax_dict['c'],
            orientation="horizontal",
            label="GFRE($X_{AniSOAP}$, $X_{SOAP}$)",
            fraction=0.6
)
cb.set_ticks([round(r, 1) for r in levels[::2]])
cb.ax.xaxis.set_label_position('top')

for al in ax_list[-1]:
    if al!='c':
        ax_dict[al].set_xlabel(r"$\sigma_1 = \sigma_2 \quad [\AA]$")

for a in ax_list:
    if a[0]!='c':
        ax_dict[a[0]].set_ylabel(r"$\sigma_3\quad  [\AA]$")
        ax_dict[a[0]].set_ylabel(r"$\sigma_3\quad  [\AA]$")

ax_dict["2.0"].set_xlim([0.45, 5.55])
ax_dict["2.0"].set_ylim([0.45, 5.55])
plt.savefig("../../../figures/si/si_hyperparam_tuning.pdf", pad_inches=0.1, bbox_inches="tight")
plt.show()

if os.path.exists("optimized_gfres.npz"):
    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(5, 5), gridspec_kw=dict(width_ratios=(1, 0.05))
    )
    g = float(np.load("optimized_gfres.npz")["optimized_gaussian"])
    x = []
    y = []
    mgres = []

    for key in dump_dict:
        a1, a3, s = read_key(key)
        if s==g:
            x.append(a1)
            y.append(a3)
            mgres.append(float(dump_dict[key]))

    if len(x) > 3:

        x = np.array(x)
        y = np.array(y)
        mgres = np.array(mgres)
        triang = tri.Triangulation(
            x,
            y,
        )

        ax.tricontour(
            triang, mgres, cmap=cmap, vmin=vmin, vmax=vmax, zorder=-2, levels=levels
        )

        p = ax.scatter(
            x,
            y,
            c=mgres,
            vmin=vmin - dl / 2.0,
            vmax=vmax + dl / 2.0,
            cmap=cmap,
            marker="o",
            s=10,  # facecolors='none',
            rasterized=True,
        )

        cb = plt.colorbar(
            p,
            ax=ax,
            cax=cax,
            orientation="vertical",
            label="GFRE($X_{AniSOAP}$, $X_{SOAP}$)",
        )
        cb.set_ticks([round(r, 1) for r in levels[::2]])

        ax.set_xlabel(r"$\sigma_1 = \sigma_2 \quad [\AA]$")
        ax.set_ylabel(r"$\sigma_3\quad  [\AA]$")

    ax.set_xlim([0.0, 5.525])
    ax.set_ylim([0.0, 5.525])
    plt.subplots_adjust(wspace=0.1)
    plt.savefig("../../../figures/hyperparam_tuning.pdf", pad_inches=0.0, bbox_inches="tight")
    plt.show()