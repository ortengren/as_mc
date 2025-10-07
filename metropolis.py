import numpy as np
import random
from trial_moves import simultaneous_move
from potentials import get_rand_energy, gay_berne_uniaxial


BOLTZCONST = 8.617333262e-5 # eV / K
TEMP = 300 # K
BETA = 1 / (BOLTZCONST * TEMP)


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    return r < np.exp(-BETA * (new_energy - old_energy))


class MetropolisCalculator:
    def __init__(self, init_frame, energy_func="random", seed=1234, pos_delt=2, or_delt=0.1):
        self.init_frame = init_frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.seed = seed
        self.frames = [init_frame]
        self.energies = [self.calc_energy(init_frame)]
        if self.seed is not None:
            self.rand = random.Random(self.seed)
        else:
            self.rand = None

    def calc_energy(self, frame, seed=None):
        if self.energy_func == "random":
            if self.rand is not None:
                return get_rand_energy(frame, self.energies[-1], self.rand)
            else:
                return get_rand_energy(frame, self.energies[-1])
        elif self.energy_func == "GB":
            r_ij = frame.get_distance(0, 1, mic=True, vector=True)
            u_i_hat = frame.arrays["or_vec"][0]
            u_j_hat = frame.arrays["or_vec"][1]
            return gay_berne_uniaxial(r_ij, u_i_hat, u_j_hat)
        else:
            return NotImplementedError()

    def step(self):
        # calculate a possible new state and calculate its energy
        trial_frame = simultaneous_move(self.frames[-1], self.pos_delt, self.or_delt)
        trial_energy = self.calc_energy(trial_frame)
        # decide whether trajectory will assume the new state or retain its current state
        keep_move = decide_accept(self.energies[-1], trial_energy)
        if keep_move:
            # add trial state to trajectory
            self.frames.append(trial_frame)
            self.energies.append(trial_energy)
        else:
            # add a copy of the current state to the trajectory
            self.frames.append(self.frames[-1])
            self.energies.append(self.energies[-1])
        # update object's step_count variable
        self.step_count += 1

    def calculate_trajectory(self, num_steps):
        while self.step_count < num_steps:
            self.step()
            # print progress of simulation
            if self.step_count % 100 == 0:
                print(self.step_count, " / ", num_steps)
            
