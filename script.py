import pickle
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from metropolis import MetropolisCalculator


with open("simulations/walsh_potential/run1/MetropolisCalculator.pkl", "rb") as f:
    metro = pickle.load(f)
# plot average move acceptance rate
decision_avgs = []
final_n = -1
n = 1
decisions = [dec.value for dec in metro.decisions]
while n < len(decisions[1:]) / 50:
    start = (n-1)*50 + 1
    stop = n*50 + 1
    avg = np.mean(decisions[start:stop])
    decision_avgs.append(avg)
    final_n = n
    n += 1
decision_avgs.append(np.mean(decisions[final_n*50 + 1:]))
decision_avgs = np.array(decision_avgs)
plt.plot(decision_avgs)
plt.savefig(f"simulations/walsh_potential/run1/decision_avgs.png")
plt.close()