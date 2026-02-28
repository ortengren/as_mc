import numpy as np
from tqdm.auto import tqdm
from trial_moves import calc_or_vec, calculate_com_move, calculate_quat_move
from potentials import gb, quadrupole, GB_PARAMS, QQ
import random
import ase
from ase.neighborlist import NeighborList
from ase.io import Trajectory
import datetime
import os


BOLTZCONST = 8.617E-5 # eV / K
TEMP = 50 # K
BETA = 1 / (BOLTZCONST * TEMP)

TARGET_ACC_RATE = 0.275


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
            pos_delt=0.15,
            or_delt=0.05,
            nl_radius=15,
            nl_skin=0.3,
            output_dir=None,
    ):
        # create new frame so .traj file correctly records array data
        frame = ase.Atoms(
            positions=init_frame.positions,
            cell=init_frame.cell,
            pbc=True,
            info=init_frame.info,
        )
        frame.new_array("c_q", init_frame.arrays["c_q"])
        frame.new_array("or_vec", init_frame.arrays["or_vec"])
        self.current_frame = frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.pos_decisions = [-1]
        self.or_decisions = [-1]
        self.current_frame.info["pos_acc_rate"] = -1
        self.current_frame.info["or_dec_rate"] = -1
        self.current_frame.info["or_delta"] = -1
        self.current_frame.info["pos_delta"] = -1
        self.equilibrated = False
        # auto generate traj file name if needed
        if output_dir is None:
            self.output_dir = "simulations/" + generate_simulation_id()
        else:
            self.output_dir = output_dir
        # set up neighborlist
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

        # determine whether to perturb position or orientation
        move_type = random.randint(0, 1)
        if move_type == 0:    # perturb position
            # record original position
            old_pos = self.current_frame.positions[rand_idx].copy()
            # get trial move
            new_pos = calculate_com_move(self.current_frame.positions[rand_idx], self.pos_delt)
            # apply trial move
            self.current_frame.positions[rand_idx] = new_pos
            self.nl.update(self.current_frame)
            # calculate new energy
            new_energy = self.calc_energy(rand_idx)
            # decide whether to accept trial move
            keep_move = decide_accept(old_energy, new_energy)
            if keep_move:
                self.pos_decisions.append(1)
            else:
                self.pos_decisions.append(0)
                self.current_frame.positions[rand_idx] = old_pos
                self.nl.update(self.current_frame)
        else:                 # perturb orientation
            # record original orientation
            old_quat = self.current_frame.arrays["c_q"][rand_idx].copy()
            old_or_vec = self.current_frame.arrays["or_vec"][rand_idx].copy()
            # get trial move
            new_quat = calculate_quat_move(self.current_frame.arrays["c_q"][rand_idx], self.or_delt)
            new_or_vec = np.squeeze(calc_or_vec(new_quat))
            # apply trial move
            self.current_frame.arrays["c_q"][rand_idx] = new_quat
            self.current_frame.arrays["or_vec"][rand_idx] = new_or_vec
            # calculate new energy
            new_energy = self.calc_energy(rand_idx)
            # decide whether to accept trial move
            keep_move = decide_accept(old_energy, new_energy)
            if keep_move:
                self.or_decisions.append(1)
            else:
                self.or_decisions.append(0)
                self.current_frame.arrays["c_q"][rand_idx] = old_quat
                self.current_frame.arrays["or_vec"][rand_idx] = old_or_vec
        self.step_count += 1

    def block_update(self, window, traj, dynamic_delta):
        # record acceptance rates for most recent block
        if len(self.pos_decisions) < window:
            pos_acc_rate = np.mean(self.pos_decisions[1:])
        else:
            pos_acc_rate = np.mean(self.pos_decisions[-window:])
        if len(self.or_decisions) < window:
            or_acc_rate = np.mean(self.or_decisions[1:])
        else:
            or_acc_rate = np.mean(self.or_decisions[-window:])
        # update frame info
        self.current_frame.info["pos_acc_rate"] = pos_acc_rate
        self.current_frame.info["or_acc_rate"] = or_acc_rate
        self.current_frame.info["step"] = self.step_count
        self.current_frame.info["pos_delta"] = self.pos_delt
        self.current_frame.info["or_delta"] = self.or_delt
        # write trajectory file
        traj.write(self.current_frame)
        # update trial move magnitude if enabled
        if dynamic_delta:
            # update position delta
            if pos_acc_rate > 0.35:
                scale_amt = min((1.1, pos_acc_rate / TARGET_ACC_RATE))
                self.pos_delt *= scale_amt
            elif pos_acc_rate < 0.20:
                scale_amt = max((0.9, pos_acc_rate / TARGET_ACC_RATE))
                self.pos_delt *= scale_amt
            # update orientation delta
            if or_acc_rate > 0.35:
                scale_amt = min((1.1, or_acc_rate / TARGET_ACC_RATE))
                self.or_delt *= scale_amt
            elif or_acc_rate < 0.20:
                scale_amt = max((0.9, or_acc_rate / TARGET_ACC_RATE))
                self.or_delt *= scale_amt

    def equilibrate(self, num_steps, block_size, dynamic_delta=True):
        # initialize Trajectory object
        traj = Trajectory(self.output_dir + "/equilibration.traj", "w", self.current_frame)
        window = block_size // 2
        with tqdm(total=num_steps, initial=self.step_count, desc="Equilibrating") as pbar:
            while self.step_count < num_steps:
                self.step(pbar)
                pbar.update(1)
                if self.step_count % block_size == 0:
                    self.block_update(window, traj, dynamic_delta)
        # close trajectory file
        traj.close()
        # update state
        self.equilibrated = True
        self.step_count = 0
        self.current_frame.info["step"] = 0

    def calculate_trajectory(self, num_steps, block_size=100, num_eq_steps=5000):
        # equilibrate
        if num_eq_steps is not None:
            self.equilibrate(num_eq_steps, block_size)
        # initialize Trajectory object
        traj = Trajectory(self.output_dir + "/simulation.traj", "w", self.current_frame)
        window = block_size // 2
        with tqdm(total=num_steps, initial=self.step_count, desc="Simulating") as pbar:
            while self.step_count < num_steps:
                self.step(pbar)
                pbar.update(1)
                if self.step_count % block_size == 0:
                    self.block_update(window, traj, dynamic_delta=False)
        # close trajectory file
        traj.close()
            

def calc_free_energy(params):
    # TODO: implement
    # define order parameters (interparticle distance & misalignment angle)
    # calc prob of each state
    # calc entropy to get density of states
    # calc partition function as -kTln(sum_{states i} p_i \Omega e^{-beta*U_i})
    return NotImplementedError