import numpy as np
import ase.io
import ase.neighborlist as nl
import networkx as nx
from ase import Atoms
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import ClebschGordanReal
from anisoap.asecg import CGRep as cg
import datetime


def separate_mols(frame):
    # build graph
    G = nx.Graph(nl.build_neighbor_list(frame).get_connectivity_matrix(sparse=False))
    separate_mols = nx.connected_components(G)
    return [cg.CGInfo(list(comp), "pentacene", "PC") for comp in separate_mols]


def get_cg_frame(frame):
    c_diameters1 = []
    c_diameters2 = []
    c_diameters3 = []
    cg_infos = separate_mols(frame)
    beads = []
    for bead in cg_infos:
        com = cg.get_center_of_mass(frame[bead.cg_indices])
        axes, quat = cg.get_quat_and_semiaxes(frame[bead.cg_indices])
        c_diameters1.append(axes[0])
        c_diameters2.append(axes[1])
        c_diameters3.append(axes[2])
        bead_frame = Atoms(positions=np.reshape(com, (1, -1)), cell=frame.cell, pbc=frame.pbc)
        bead_frame.arrays["quats"] = np.reshape(quat, (1, -1))
        bead_frame.arrays["c_diameter[1]"] = adjust_c_diameter(axes[0], 1, 12)
        bead_frame.arrays["c_diameter[2]"] = adjust_c_diameter(axes[1], 1, 12)
        bead_frame.arrays["c_diameter[3]"] = adjust_c_diameter(axes[2], 1, 12)
        beads.append(bead_frame)
    cg_frame = beads[0]
    for i in range(1, len(beads)):
        cg_frame.extend(beads[i])
    return cg_frame


def adjust_c_diameter(val, m, M):
    val = np.max((val, m))
    val = np.min((val, M))
    return val


def get_ell_frames(frames, write_file=False):
    ell_frames = []
    for frame in frames:
        ell_frames.append(get_cg_frame(frame))
    if write_file:
        dt = datetime.datetime.today().isoformat(timespec="minutes")
        fname = "output/pentacenes_cg_" + dt + ".xyz"
        ase.io.write(fname, ell_frames)
    return ell_frames


if __name__ == "__main__":
    data_path = "Pentacene/Pentacene/traj-pentacene.xyz"
    frames = ase.io.read(data_path, ":")
    print(len(frames))

    ell_frames = get_ell_frames(frames, write_file=True)