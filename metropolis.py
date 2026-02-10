import numpy as np
import random
from trial_moves import simultaneous_move
from potentials import calc_walsh_potential, gb, GB_PARAMS
from enum import Enum
import random


BOLTZCONST = 1.9872041e-3 # kcal / mol*K
TEMP = 200 # K
BETA = 1 / (BOLTZCONST * TEMP)


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    dec_term = np.exp(-BETA * (new_energy - old_energy))
    decision = r < dec_term
    return decision


class MetropolisCalculator:
    def __init__(self, init_frame, energy_func="GB", pos_delt=0.2, or_delt=0.2):
        self.init_frame = init_frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.frames = [init_frame]
        self.decisions = [-1]

    def calc_energy(self, frame, idx):
        if self.energy_func == "Walsh":
            return calc_walsh_potential(frame, idx)
        elif self.energy_func == "GB":
            U_GB = 0
            uhat1 = frame.arrays["or_vec"][idx]
            for i in range(len(frame)):
                if i == idx:
                    continue
                r = frame.get_distance(idx, i, mic=True, vector=True)
                if np.linalg.norm(r) > 20:
                    continue
                uhat2 = frame.arrays["or_vec"][i]
                U_GB += gb(uhat1, uhat2, r, *GB_PARAMS.values())
            return U_GB
        else:
            return NotImplementedError()

    def step(self):
        # choose particle to update
        num_particles = len(self.frames[-1])
        rand_idx = random.randint(0, num_particles - 1)
        # calculate particle's contribution to total energy
        old_energy = self.calc_energy(self.frames[-1], rand_idx)
        # calculate a possible new state and calculate its energy
        trial_frame = simultaneous_move(self.frames[-1], rand_idx, self.pos_delt, self.or_delt)
        trial_energy = self.calc_energy(trial_frame, rand_idx)
        # decide whether trajectory will assume the new state or retain its current state
        keep_move = decide_accept(old_energy, trial_energy)
        if keep_move:
            # add trial state to trajectory
            self.frames.append(trial_frame)
            self.decisions.append(1)
        else:
            # add a copy of the current state to the trajectory
            self.frames.append(self.frames[-1])
            self.decisions.append(0)
        # update object's step_count variable
        self.step_count += 1

    def calculate_trajectory(self, num_steps):
        while self.step_count < num_steps:
            self.step()
            # print progress of simulation
            # TODO: use tqdm instead of printing
            if self.step_count % 100 == 0:
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
            

def calc_free_energy(params):
    # TODO: implement
    # define order parameters (interparticle distance & misalignment angle)
    # calc prob of each state
    # calc entropy to get density of states
    # calc partition function as -kTln(sum_{states i} p_i \Omega e^{-beta*U_i})
    return NotImplementedError