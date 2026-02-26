import numpy as np
import random

from pygments.lexers import oldmod

from trial_moves import calc_or_vec, calculate_com_move, calculate_quat_move
from potentials import gb, quadrupole, GB_PARAMS, QQ
from enum import Enum
import random
from ase.neighborlist import NeighborList
from ase.io import Trajectory
import datetime
import os


BOLTZCONST = 8.617E-5 # eV / K
TEMP = 100 # K
BETA = 1 / (BOLTZCONST * TEMP)


def generate_simulation_id(method="datetime"):
    if method == "datetime":
        dt = datetime.datetime.today().isoformat(timespec="minutes")
        if not os.path.exists(f"simulations/{dt}"):
            os.makedirs(f"simulations/{dt}")
        return dt
    else:
        return NotImplementedError


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    dec_term = np.exp(-BETA * (new_energy - old_energy))
    decision = r < dec_term
    return decision


class MetropolisCalculator:
    def __init__(
            self,
            init_frame,
            energy_func="GB",
            pos_delt=0.2,
            or_delt=0.2,
            nl_radius=15,
            nl_skin=0.3,
            traj_file=None,
    ):
        self.current_frame = init_frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.decisions = [-1]
        # auto generate traj file name if needed
        if traj_file is None:
            self.traj_file = generate_simulation_id()
        # initialize Trajectory object
        self.traj = Trajectory(self.traj_file, "w", self.current_frame)
        cutoffs = [nl_radius / 2] * len(self.current_frame)
        self.nl = NeighborList(
            cutoffs,
            skin=nl_skin,
            sorted=False,
            self_interaction=False,
            bothways=True,
        )
        self.nl.update(self.current_frame)

    def calc_energy(self, center_idx):
        if self.energy_func == "GB":
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
            uhat1 = np.array([center_ell_orientation for _ in range(len(indices))])
            assert np.shape(uhat1) == np.shape(uhat2), \
                f"uhat1 and uhat2 are different shapes \n{np.shape(uhat1)} != {np.shape(uhat2)}"
            # calculate pairwise interaction energies of center particle w/ each neighbor
            gb_e = gb(uhat1, uhat2, displacements, *GB_PARAMS.values())
            qq_e = np.squeeze(quadrupole(uhat1, uhat2, displacements, QQ))
            pw_energies = gb_e + qq_e
            # total particle energy contribution is sum over pairs
            energy = np.sum(pw_energies)
            return energy
        else:
            return NotImplementedError()

    def step(self):
        # choose particle to update
        num_particles = len(self.current_frame)
        rand_idx = random.randint(0, num_particles - 1)
        # calculate particle's contribution to total energy
        old_energy = self.calc_energy(rand_idx)
        # store original position and orientation
        old_pos = self.current_frame.positions[rand_idx].copy()
        old_quat = self.current_frame.arrays["or_vec"][rand_idx].copy()
        old_or_vec = self.current_frame.arrays["or_vec"][rand_idx].copy()
        # get trial move
        new_pos = calculate_com_move(self.current_frame.positions[rand_idx], self.pos_delt)
        new_quat = calculate_quat_move(self.current_frame.arrays["c_q"][rand_idx], self.or_delt)
        new_or_vec = calc_or_vec(new_quat)
        # apply trial move
        self.current_frame.positions[rand_idx] = new_pos
        self.current_frame.arrays["c_q"][rand_idx] = new_quat
        self.current_frame.arrays["or_vec"][rand_idx] = new_or_vec
        self.nl.update(self.current_frame)
        # calculate new energy
        new_energy = self.calc_energy(rand_idx)
        # decide whether simulation will accept trial move
        keep_move = decide_accept(old_energy, new_energy)
        if keep_move:
            # retain current state
            self.decisions.append(1)
        else:
            # revert to previous state and update neighborlist
            self.current_frame.positions[rand_idx] = old_pos
            self.current_frame.arrays["c_q"][rand_idx] = old_quat
            self.current_frame.arrays["or_vec"][rand_idx] = old_or_vec
            self.nl.update(self.current_frame)
            self.decisions.append(0)
        self.step_count += 1

    def calculate_trajectory(self, num_steps):
        while self.step_count < num_steps:
            self.step()
            # print progress of simulation
            # TODO: use tqdm instead of printing
            if self.step_count % 100 == 0:
                self.traj.write(self.current_frame)
                print(self.step_count, " / ", num_steps)
                acc_rate = np.mean(self.decisions[self.step_count-99:self.step_count+1])
                print(f"acc_rate: {acc_rate}")
                if acc_rate > 0.30:
                    self.or_delt += 0.1
                    self.pos_delt += 0.1
                    print(f"or_delt: {self.or_delt-0.1} -> {self.or_delt}")
                    print(f"pos_delt: {self.pos_delt-0.1} -> {self.pos_delt}")
                if acc_rate < 0.30:
                    self.or_delt -= 0.05
                    self.pos_delt -= 0.05
                    print(f"or_delt: {self.or_delt+0.05} -> {self.or_delt}")
                    print(f"pos_delt: {self.pos_delt+0.05} -> {self.pos_delt}")
        # close trajectory file
        self.traj.close()
            

def calc_free_energy(params):
    # TODO: implement
    # define order parameters (interparticle distance & misalignment angle)
    # calc prob of each state
    # calc entropy to get density of states
    # calc partition function as -kTln(sum_{states i} p_i \Omega e^{-beta*U_i})
    return NotImplementedError