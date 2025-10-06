def main():
    import ase.io
    from metropolis import MetropolisCalculator
    import ase
    import pickle

    frames = ase.io.read("two_benzenes.xyz", ":")
    ell_frames = ase.io.read("two_ellipsoids.xyz", ":")
    for ell_frame, frame in zip(ell_frames, frames):
        ell_frame.info["energy"] = frame.get_total_energy()

    metro = MetropolisCalculator(ell_frames[0], ell_frames[0].info["energy"])
    metro.calculate_trajectory(1_000)

    pickle.dump(metro, "metro_9-29-25_2:05.pickle")
    print("done")


if __name__ == "__main__":
    main()
