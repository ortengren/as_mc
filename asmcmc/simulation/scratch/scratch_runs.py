from asmcmc.simulation.run import equilibrate_grid, equilibrate_point


def main():
    equilibrate_grid(
        temps=(
            100.0,
            200.0,
        ),
        pressures=(1e-5, 5e-5),
        out_dir="results/simulations/eq_grid",
        num_eq_steps=200_000,
        block_size=250,
        buffer_size=20,
        nl_radius=10.0,
        nl_skin=2.0,
        npt_ensemble=True,
        n_particles=100,
        seed=None,
        progress=True,
    )


if __name__ == "__main__":
    main()
