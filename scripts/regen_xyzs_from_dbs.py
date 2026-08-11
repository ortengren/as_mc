from ase.db import connect
import ase.io
import numpy as np
from tqdm import tqdm

names = [
    "100.0_6.324209e-07/herringbone",
    "100.0_6.324209e-07/herringbone_jittered",
    "100.0_6.324209e-07/herringbone_jittered_2",
    "300.0_6.324209e-07/herringbone_jittered_0",
]

for name in names:
    db_path = f"../results/validation/{name}/simulation.db"
    xyz_path = f"../results/validation/{name}/simulation.xyz"
    traj = []
    with connect(db_path) as db:
        for row in tqdm(
            db.select(), total=db.count(), desc=name, unit="frame", leave=False
        ):
            atoms = row.toatoms()
            atoms.new_array("c_q", np.asarray(row.data["c_q"]))
            atoms.new_array("or_vec", np.asarray(row.data["or_vec"]))
            atoms.new_array("shape", np.asarray([[2.5, 2.5, 1.0] for _ in atoms]))
            atoms.info["total_energy"] = row["total_energy"]

            traj.append(atoms)

        ase.io.write(xyz_path, traj, format="extxyz")


for name in names:
    db_path = f"../results/validation/{name}/equilibration.db"
    xyz_path = f"../results/validation/{name}/equilibration.xyz"
    traj = []
    with connect(db_path) as db:
        for row in tqdm(
            db.select(), total=db.count(), desc=name, unit="frame", leave=False
        ):
            atoms = row.toatoms()
            atoms.new_array("c_q", np.asarray(row.data["c_q"]))
            atoms.new_array("or_vec", np.asarray(row.data["or_vec"]))
            atoms.new_array("shape", np.asarray([[2.5, 2.5, 1.0] for _ in atoms]))
            atoms.info["total_energy"] = row["total_energy"]

            traj.append(atoms)

        ase.io.write(xyz_path, traj, format="extxyz")
