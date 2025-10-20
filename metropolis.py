import numpy as np
import random
from trial_moves import simultaneous_move
from potentials import get_rand_energy, gay_berne_walsh
from enum import Enum


BOLTZCONST = 8.617333262e-5 # eV / K
TEMP = 300 # K
BETA = 1 / (BOLTZCONST * TEMP)


class Decision(Enum):
    """
    Enum for tracking whether a trial move is accepted or rejected.
    """
    ACCEPT = 1
    REJECT = 0
    INIT = -1


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    return r < np.exp(-BETA * (new_energy - old_energy))


class MetropolisCalculator:
    def __init__(self, init_frame, energy_func="GB", seed=None, pos_delt=0.02, or_delt=0.001):
        self.init_frame = init_frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.seed = seed
        self.frames = [init_frame]
        self.energies = [self.calc_energy(init_frame)]
        self.decisions = [Decision.INIT]
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
            return gay_berne_walsh(frame)
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
            self.decisions.append(Decision.ACCEPT)
        else:
            # add a copy of the current state to the trajectory
            self.frames.append(self.frames[-1])
            self.energies.append(self.energies[-1])
            self.decisions.append(Decision.REJECT)
        # update object's step_count variable
        self.step_count += 1

    def calculate_trajectory(self, num_steps):
        while self.step_count < num_steps:
            self.step()
            # print progress of simulation
            # TODO: use tqdm instead of printing
            if self.step_count % 100 == 0:
                print(self.step_count, " / ", num_steps)

    def get_acc_rate(self, at_step=None):
        if at_step is None:
            at_step = self.step_count
        num_acc = 0
        for dec in self.decisions[1 : at_step+1:]:
            num_acc += dec.value
        return num_acc / at_step
            

def calc_free_energy(params):
    # TODO: implement
    # define order parameters (interparticle distance & misalignment angle)
    # calc prob of each state
    # calc entropy to get density of states
    # calc partition function as -kTln(sum_{states i} p_i \Omega e^{-beta*U_i})
    return NotImplementedError