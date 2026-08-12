import signac
import itertools

project = signac.get_project()

# Parameters that I think will be swept
T_arr = [250, 273.15, 300]
P_arr = [0.75, 1.0, 1.5]    # atm
potential_arr = ['anisoap', 'gbq']
initial_config_arr = ['herringbone']

# Parameters that probably won't be swept, but retain for explicitly denoting simulation conditions.
pos_jitter_arr = [0.15]
or_jitter_arr = [0.15]
vol_delta_arr = [0.025]
seed_arr = [45]
n_particles_arr = [400]
nl_radius_arr = [6.8]
nl_skin_arr = [1.0]
max_or_delt_arr = [0.25]

all_combos = list(itertools.product(T_arr, P_arr, potential_arr, initial_config_arr, pos_jitter_arr, or_jitter_arr, vol_delta_arr, seed_arr, n_particles_arr, nl_radius_arr, nl_skin_arr, max_or_delt_arr))
print(f"{len(all_combos)=}")

for T, P, potential, initial_config, pos_jitter, or_jitter, vol_delta, seed, n_particles, nl_radius, nl_skin, max_or_delt in all_combos:
    project.open_job(
        {
            "T":T,
            "P":P,
            "potential":potential,
            "initial_config":initial_config,
            "pos_jitter":pos_jitter,
            "or_jitter":or_jitter,
            "vol_delta":vol_delta,
            "seed":seed,
            "n_particles":n_particles,
            "nl_radius":nl_radius,
            "nl_skin":nl_skin,
            "max_or_delt":max_or_delt
        }
    ).init()