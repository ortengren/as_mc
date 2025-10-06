import numpy as np
import random
from trial_moves import simultaneous_move
from potentials import get_rand_energy


BOLTZCONST = 8.617333262e-5 # eV / K
TEMP = 300 # K
BETA = 1 / (BOLTZCONST * TEMP)


def decide_accept(old_energy, new_energy):
    r = random.uniform(0, 1)
    print(f"old energy: {old_energy}; new energy: {new_energy}")
    print(f"r: {r}; compared to: {np.exp(-BETA * (new_energy - old_energy))}")
    print(f"new energy - old energy: {new_energy-old_energy}")
    return r < np.exp(-BETA * (new_energy - old_energy))


class MetropolisCalculator:
    def __init__(self, init_frame, init_energy, energy_func="random", seed=1234, pos_delt=2, or_delt=0.1):
        self.init_frame = init_frame
        self.pos_delt = pos_delt
        self.or_delt = or_delt
        self.step_count = 0
        self.energy_func = energy_func
        self.seed = seed
        self.frames = [init_frame]
        self.energies = [init_energy]
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
        else:
            return NotImplementedError()

    def step(self):
        trial_frame = simultaneous_move(self.frames[-1], self.pos_delt, self.or_delt)
        trial_energy = self.calc_energy(trial_frame)
        print(f"trial energy: {trial_energy}")
        keep_move = decide_accept(self.energies[-1], trial_energy)
        print(f"keep move: {keep_move}")
        if keep_move:
            self.frames.append(trial_frame)
            self.energies.append(trial_energy)
            print(f"trial energy added: {trial_energy}")
        else:
            self.frames.append(self.frames[-1])
            self.energies.append(self.energies[-1])
            print(f"old energy kept: {self.energies[-1]}")
        self.step_count += 1

    def calculate_trajectory(self, num_steps):
        while self.step_count < num_steps:
            self.step()
            if self.step_count % 100 == 0:
                print(self.step_count, " / ", num_steps)
            
