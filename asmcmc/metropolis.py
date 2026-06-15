import numpy as np
from tqdm.auto import tqdm
from asmcmc.initialize import Initializer, FrameInitializer, RandomLatticeInitializer
from asmcmc.trial_moves import (
    calc_or_vec,
    calculate_com_move,
    calculate_quat_move,
    calculate_vol_move,
)
from asmcmc.potentials import Potential, calc_total_energy, DEFAULT_POTENTIAL
import random
import ase
from ase.neighborlist import NeighborList
from ase.io import Trajectory
import datetime
import os
from ase.db import connect
from typing import Optional

BOLTZCONST = 8.617e-5  # eV / K

TARGET_ACC_RATE = 0.275


def generate_simulation_id(method="datetime"):
    """Generate an ID for the simulation."""

    if method == "datetime":
        dt = datetime.datetime.today().isoformat(timespec="minutes")
        if not os.path.exists(f"results/simulations/{dt}"):
            os.makedirs(f"results/simulations/{dt}")
        return dt
    else:
        raise NotImplementedError


def npt_decide_accept(old_en, new_en, old_vol, new_vol, beta, pressure, num_part):
    """Decide whether to accept a new configuration with probability min(p, 1), where

        p = exp(-beta * (new_en - old_en + P * (new_vol - old_vol)) - (N + 1) * log(new_vol / old_vol))

    This function is suitable for the NPT ensemble.

    Pressure: eV/Å^3
    Volume: Å^3
    """
    if new_vol <= 0:
        return False
    r = random.uniform(0, 1)
    arg = -beta * (
        new_en
        - old_en
        + pressure * (new_vol - old_vol)
        - (num_part + 1) * np.log(new_vol / old_vol) / beta
    )
    if arg >= 0:
        return True  # always accept; avoids exp overflow for favorable moves
    return r < np.exp(arg)


def nvt_decide_accept(old_en, new_en, beta):
    """Decide whether to accept a new configuration with probability min(p, 1),
    where

        p = exp(-beta * (new_en - old_en))

    This function is suitable for the NVT ensemble.
    """
    r = random.uniform(0, 1)
    arg = -beta * (new_en - old_en)
    if arg >= 0:
        return True  # always accept; avoids exp overflow for favorable moves
    return r < np.exp(arg)


# TODO: Double check NVT logic
class MetropolisCalculator:
    """Main class for the Metropolis Monte Carlo simulation.  This class handles
    most of the core logic of the simulation.
    """

    def __init__(
        self,
        temp,
        pressure,
        init_frame: Optional[ase.Atoms] = None,
        initializer: Optional[Initializer] = None,
        potential: Optional[Potential] = None,
        pos_delt=0.15,
        or_delt=0.05,
        vol_delt=0.05,
        nl_radius=15,
        nl_skin=1.0,
        output_dir=None,
        npt_ensemble=True,
    ):
        # Resolve the frame source to a single Initializer. ``init_frame`` is a
        # convenience that wraps a supplied frame; ``initializer`` lets callers
        # control how the config is built; with neither, fall back to a random
        # lattice.
        if init_frame is not None and initializer is not None:
            raise ValueError("Provide at most one of init_frame and initializer.")
        if init_frame is not None:
            initializer = FrameInitializer(init_frame)
        elif initializer is None:
            initializer = RandomLatticeInitializer()
        self.initializer = initializer
        source = initializer.generate()
        self.init_frame = source

        # Copy the initial frame to ensure that it is not mutated by the simulation
        frame = ase.Atoms(
            positions=source.positions.copy(),
            cell=source.cell.copy(),
            pbc=True,
            info=source.info.copy(),
        )
        frame.new_array("c_q", source.arrays["c_q"].copy())
        frame.new_array("or_vec", source.arrays["or_vec"].copy())
        self.current_frame = frame

        # Initialize the simulation parameters
        self.beta = 1 / (temp * BOLTZCONST)
        self.pressure = pressure
        self.current_vol = np.linalg.det(frame.cell)
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.vol_delt = vol_delt
        self.step_count = 0
        self.potential: Potential = (
            potential if potential is not None else DEFAULT_POTENTIAL
        )
        self.pos_decisions = []
        self.or_decisions = []
        self.vol_decisions = []
        self.current_frame.info["pos_acc_rate"] = -1
        self.current_frame.info["or_acc_rate"] = -1
        self.current_frame.info["or_delta"] = -1
        self.current_frame.info["pos_delta"] = -1
        self.current_frame.info["potential"] = self.potential.name
        self.equilibrated = False

        # whether to attempt volume moves (NPT); set False for NVT
        self.npt_ensemble = npt_ensemble

        # auto generate traj file name if needed
        if output_dir is None:
            self.output_dir = f"results/simulations/{generate_simulation_id()}"
        else:
            self.output_dir = output_dir

        # set up neighborlist
        self.nl_cutoffs = [nl_radius] * len(self.current_frame)
        self.nl_skin = nl_skin
        self.nl = NeighborList(
            self.nl_cutoffs,
            skin=self.nl_skin,
            sorted=False,
            self_interaction=False,
            bothways=True,
        )
        self.nl.update(self.current_frame)
        self.current_energy = calc_total_energy(
            self.current_frame, self.nl_cutoffs, potential=self.potential
        )
        self.current_frame.info["total_energy"] = self.current_energy

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calc_energy(self, center_idx):
        """Calculate the energy of particle at index `center_idx`."""

        center_pos = self.current_frame.positions[center_idx].copy()

        # find neighbors
        indices, offsets = self.nl.get_neighbors(center_idx)

        # energy is 0 if center particle has no neighbors
        if len(indices) == 0:
            return 0

        # calculate neighbor positions while respecting pbc
        cell = self.current_frame.get_cell()
        shift_vecs = np.dot(offsets, cell)
        neighbor_positions = self.current_frame.positions[indices] + shift_vecs
        displacements = neighbor_positions - center_pos

        # find neighbor orientations
        center_ell_orientation = self.current_frame.arrays["or_vec"][center_idx].copy()
        uhat2 = self.current_frame.arrays["or_vec"][indices].copy()
        uhat1 = np.broadcast_to(center_ell_orientation, (len(indices), 3))
        assert np.shape(uhat1) == np.shape(
            uhat2
        ), f"uhat1 and uhat2 are different shapes \n{np.shape(uhat1)} != {np.shape(uhat2)}"

        # pairwise interaction energies of center particle w/ each neighbor;
        # total particle energy contribution is the sum over pairs
        pw_energies = self.potential.pair_energy(uhat1, uhat2, displacements)

        return np.sum(pw_energies)

    def step(self):
        """Performs a single step of the simulation, including performing trial
        moves, determining acceptance, and updating the state.
        """
        num_particles = len(self.current_frame)

        # determine whether to perturb volume, position, or orientation
        if self.npt_ensemble:
            r = random.uniform(0, num_particles)
            if r < (num_particles - 1) / 2:
                move_type = "position"
            elif r < num_particles - 1:
                move_type = "orientation"
            else:
                move_type = "volume"
        else:
            # NVT: attempt only translations and rotations
            move_type = "position" if random.random() < 0.5 else "orientation"

        # perform trial move
        if move_type == "volume":

            # record original values
            old_cell = self.current_frame.get_cell().copy()
            old_vol = self.current_vol.copy()
            old_total_energy = self.current_energy.copy()

            # calculate new values
            new_cell, new_vol = calculate_vol_move(old_cell, old_vol, self.vol_delt)

            # apply new values
            self.current_frame.set_cell(new_cell, scale_atoms=True)

            # update neighborlist
            self.nl.update(self.current_frame)

            # calculate new energy
            new_total_energy = calc_total_energy(
                self.current_frame, self.nl_cutoffs, potential=self.potential
            )

            # decide whether to accept trial move
            keep_move = npt_decide_accept(
                old_total_energy,
                new_total_energy,
                old_vol,
                new_vol,
                self.beta,
                self.pressure,
                num_particles,
            )
            if keep_move:
                self.vol_decisions.append(1)
                self.current_energy = new_total_energy
                self.current_vol = new_vol
                self.current_frame.info["total_energy"] = self.current_energy
            else:
                self.vol_decisions.append(0)
                self.current_frame.set_cell(old_cell, scale_atoms=True)

        elif move_type == "position":

            # choose particle to update
            rand_idx = random.randint(0, num_particles - 1)

            # calculate particle's contribution to total energy
            old_energy = self.calc_energy(rand_idx)

            # record original position
            old_pos = self.current_frame.positions[rand_idx].copy()

            # get trial move
            new_pos = calculate_com_move(
                self.current_frame.positions[rand_idx], self.pos_delt
            )

            # apply trial move
            self.current_frame.positions[rand_idx] = new_pos
            self.nl.update(self.current_frame)

            # calculate new energy
            new_energy = self.calc_energy(rand_idx)

            # decide whether to accept trial move
            if self.npt_ensemble:
                keep_move = npt_decide_accept(
                    old_energy,
                    new_energy,
                    self.current_vol,
                    self.current_vol,
                    self.beta,
                    self.pressure,
                    num_particles,
                )
            else:
                keep_move = nvt_decide_accept(
                    old_energy,
                    new_energy,
                    self.beta,
                )
            if keep_move:
                self.pos_decisions.append(1)
                energy_change = new_energy - old_energy
                self.current_energy += energy_change
                self.current_frame.info["total_energy"] = self.current_energy
            else:
                self.pos_decisions.append(0)
                self.current_frame.positions[rand_idx] = old_pos
                self.nl.update(self.current_frame)  # restore NL reference to old_pos

        elif move_type == "orientation":

            # choose particle to update
            rand_idx = random.randint(0, num_particles - 1)

            # calculate particle's contribution to total energy
            old_energy = self.calc_energy(rand_idx)

            # record original orientation
            old_quat = self.current_frame.arrays["c_q"][rand_idx].copy()
            old_or_vec = self.current_frame.arrays["or_vec"][rand_idx].copy()

            # get trial move
            new_quat = calculate_quat_move(
                self.current_frame.arrays["c_q"][rand_idx], self.or_delt
            )
            new_or_vec = np.squeeze(calc_or_vec(new_quat))

            # apply trial move
            self.current_frame.arrays["c_q"][rand_idx] = new_quat
            self.current_frame.arrays["or_vec"][rand_idx] = new_or_vec

            # calculate new energy
            new_energy = self.calc_energy(rand_idx)

            # decide whether to accept trial move
            if self.npt_ensemble:
                keep_move = npt_decide_accept(
                    old_energy,
                    new_energy,
                    self.current_vol,
                    self.current_vol,
                    self.beta,
                    self.pressure,
                    num_particles,
                )
            else:
                keep_move = nvt_decide_accept(
                    old_energy,
                    new_energy,
                    self.beta,
                )
            if keep_move:
                self.or_decisions.append(1)
                energy_change = new_energy - old_energy
                self.current_energy += energy_change
                self.current_frame.info["total_energy"] = self.current_energy
            else:
                self.or_decisions.append(0)
                self.current_frame.arrays["c_q"][rand_idx] = old_quat
                self.current_frame.arrays["or_vec"][rand_idx] = old_or_vec

        self.step_count += 1

    def block_update(
        self,
        window,
        buffer,
        db_file,
        dynamic_delta=True,
        buffer_size=100,
        max_scale=1.1,
        min_scale=0.9,
    ):
        """Perform a block update of the simulation.

        This involves wrapping particles, calculating acceptance rates,
        recording data, and writing to the database.
        """
        # wrap particles to simulation box
        self.current_frame.wrap()

        # record acceptance rates for most recent block
        if len(self.pos_decisions) < window:
            pos_acc_rate = np.mean(self.pos_decisions[1:])
        else:
            pos_acc_rate = np.mean(self.pos_decisions[-window:])
        if len(self.or_decisions) < window:
            or_acc_rate = np.mean(self.or_decisions[1:])
        else:
            or_acc_rate = np.mean(self.or_decisions[-window:])
        if not self.npt_ensemble:
            vol_acc_rate = float("nan")  # no volume moves attempted in NVT
        elif len(self.vol_decisions) < window:
            vol_acc_rate = np.mean(self.vol_decisions[1:])
        else:
            vol_acc_rate = np.mean(self.vol_decisions[-window:])

        # update frame info
        scalar_data = {
            "pos_acc_rate": pos_acc_rate,
            "or_acc_rate": or_acc_rate,
            "vol_acc_rate": vol_acc_rate,
            "step": self.step_count,
            "pos_delta": self.pos_delt,
            "or_delta": self.or_delt,
            "vol_delta": self.vol_delt,
            "total_energy": self.current_energy,
            "num_particles": len(self.current_frame),
            "vol": self.current_vol,
            "potential": self.potential.name,
        }
        array_data = {
            "c_q": self.current_frame.arrays["c_q"].copy(),
            "or_vec": self.current_frame.arrays["or_vec"].copy(),
            "cell": self.current_frame.get_cell().copy(),
        }

        # Add data to buffer, which is just a list kept in memory.  When this
        # list contains `buffer_size` or more items, each item in the buffer
        # is written to the database and the buffer is cleared.  This is just a
        # simple way to avoid the cost of writing to the database on every step.
        buffer.append((self.current_frame.copy(), scalar_data, array_data))
        if len(buffer) >= buffer_size:
            # write to database
            with connect(db_file) as db:
                for triplet in buffer:
                    db.write(triplet[0], key_value_pairs=triplet[1], data=triplet[2])
            # clear buffer
            buffer.clear()

        # update trial move magnitude if enabled
        if dynamic_delta:
            # update position delta
            if pos_acc_rate > 0.35:
                scale_amt = min((max_scale, pos_acc_rate / TARGET_ACC_RATE))
                self.pos_delt *= scale_amt
            elif pos_acc_rate < 0.20:
                scale_amt = max((min_scale, pos_acc_rate / TARGET_ACC_RATE))
                self.pos_delt *= scale_amt
            self.pos_delt = min(self.pos_delt, 2.0)  # cap to prevent unbounded growth

            # update orientation delta
            if or_acc_rate > 0.35:
                scale_amt = min((max_scale, or_acc_rate / TARGET_ACC_RATE))
                self.or_delt *= scale_amt
            elif or_acc_rate < 0.20:
                scale_amt = max((min_scale, or_acc_rate / TARGET_ACC_RATE))
                self.or_delt *= scale_amt

            # update volume delta (NPT only)
            if self.npt_ensemble:
                if vol_acc_rate > 0.35:
                    scale_amt = min((max_scale, vol_acc_rate / TARGET_ACC_RATE))
                    self.vol_delt *= scale_amt
                elif vol_acc_rate < 0.20:
                    scale_amt = max((min_scale, vol_acc_rate / TARGET_ACC_RATE))
                    self.vol_delt *= scale_amt
                self.vol_delt = min(self.vol_delt, 0.5)  # cap to keep s_v positive
        return buffer

    def equilibrate(
        self,
        num_steps,
        block_size,
        dynamic_delta=True,
        buffer_size=100,
        progress=True,
        max_scale=1.1,
        min_scale=0.9,
    ):
        """Perform an equilibration of the simulation."""

        window = block_size // 2

        # initialize buffer
        buffer = []
        db_file = self.output_dir + "/equilibration.db"

        # run simulation
        with tqdm(
            total=num_steps,
            initial=self.step_count,
            desc="Equilibrating",
            disable=not progress,
        ) as pbar:
            while self.step_count < num_steps:
                self.step()
                pbar.update(1)
                if self.step_count % block_size == 0:
                    buffer = self.block_update(
                        window,
                        buffer,
                        db_file,
                        dynamic_delta=dynamic_delta,
                        buffer_size=buffer_size,
                        max_scale=max_scale,
                        min_scale=min_scale,
                    )

        # update state
        self.equilibrated = True
        self.step_count = 0
        self.current_frame.info["step"] = 0

        # write any frames left in buffer
        if len(buffer) > 0:
            with connect(db_file) as db:
                for triplet in buffer:
                    db.write(triplet[0], key_value_pairs=triplet[1], data=triplet[2])

    def calculate_trajectory(
        self,
        num_steps,
        block_size=100,
        num_eq_steps=10_000,
        buffer_size=100,
        eq_block_size=None,
        max_scale=1.1,
        min_scale=0.9,
        progress=True,
    ):
        """Performs a simulation of the system.  This method will first equilibrate
        the system, then perform the main simulation.
        """
        # equilibrate
        if eq_block_size is None:
            eq_block_size = block_size
        if num_eq_steps is not None:
            self.equilibrate(
                num_eq_steps,
                eq_block_size,
                buffer_size=buffer_size,
                max_scale=max_scale,
                min_scale=min_scale,
                progress=progress,
            )
            self.pos_decisions = []
            self.or_decisions = []
            self.vol_decisions = []
        window = block_size // 2

        # initialize buffer
        buffer = []

        # initialize database
        db_file = self.output_dir + "/simulation.db"
        with tqdm(
            total=num_steps,
            initial=self.step_count,
            desc="Simulating",
            disable=not progress,
        ) as pbar:
            while self.step_count < num_steps:
                self.step()
                pbar.update(1)
                if self.step_count % block_size == 0:
                    buffer = self.block_update(
                        window,
                        buffer,
                        db_file,
                        dynamic_delta=False,
                        buffer_size=buffer_size,
                    )

            # write any frames left in buffer
            if len(buffer) > 0:
                with connect(db_file) as db:
                    for triplet in buffer:
                        db.write(
                            triplet[0], key_value_pairs=triplet[1], data=triplet[2]
                        )
