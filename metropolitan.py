import numpy as np
import random
from trial_moves import simultaneous_move


BOLTZCONST = 8.617333262e-5 # eV / K
TEMP = 300 # K
BETA = 1 / (BOLTZCONST * TEMP)


def calc_energy(frame):
    raise NotImplementedError(
        "Need to train model to calculate energy"
    )


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    return r < np.exp(-BETA * (new_energy - old_energy))


class MetropolitanCalculator:
    def __init__(self, init_frame, pos_delt, or_delt):
        self.init_frame = init_frame
        self.frames = [(init_frame, calc_energy(init_frame))]
        self.pos_delt = pos_delt
        self.or_delt = or_delt

    def step(self):
        trial_frame = simultaneous_move(self.frames[-1], self.pos_delt, self.or_delt)
        trial_energy = calc_energy(trial_frame)
        if decide_accept(self.frames[-1][1], trial_energy):
            self.frames.append((trial_frame, trial_energy))
        else:
            self.frames.append(self.frames[-1])
