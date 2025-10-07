def main():
    import ase.io
    from metropolis import MetropolisCalculator
    import ase
    import pickle
    import datetime

    frames = ase.io.read("two_benzenes.xyz", ":")
    ell_frames = ase.io.read("two_ells_with_axes.xyz", ":")
    for ell_frame, frame in zip(ell_frames, frames):
        ell_frame.info["energy"] = frame.get_total_energy()

    metro = MetropolisCalculator(ell_frames[0], ell_frames[0].info["energy"], energy_func="GB")
    metro.calculate_trajectory(10_000)

    dt = datetime.datetime.today().isoformat(timespec="minutes")
    with open(f"metro_{dt}.pkl", "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(metro, f)
    print("done")


if __name__ == "__main__":
    main()
