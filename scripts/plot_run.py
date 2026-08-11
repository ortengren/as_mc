"""
plot_run.py — Render diagnostics for one run directory.

Thin CLI over :mod:`asmcmc.diagnostics`; the plotting itself lives in the package
so a notebook can import it without an argparse entry point.

    python scripts/plot_run.py RUN_DIR                       # all four figures
    python scripts/plot_run.py RUN_DIR --phase --structure   # just those two
    python scripts/plot_run.py RUN_DIR --db simulation.db    # the production run

Figures are written into RUN_DIR, prefixed with the db stem, so rendering a
production trajectory never clobbers the equilibration figures.
"""

import argparse

from asmcmc.diagnostics import PLOTS, render


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="Run directory holding the db.")
    parser.add_argument(
        "--db",
        default="equilibration.db",
        help="Which db to read (default: equilibration.db; use simulation.db for production).",
    )
    parser.add_argument(
        "--out-dir", default=None, help="Where to write the PNGs (default: RUN_DIR)."
    )
    for name in PLOTS:
        parser.add_argument(
            f"--{name}", action="store_true", help=f"Render {name}.png."
        )
    args = parser.parse_args(argv)

    # No plot flags at all means "everything" -- the common case is looking at a
    # run you just finished, not picking one panel.
    selected = [name for name in PLOTS if getattr(args, name)] or None

    written = render(
        args.run_dir, which=selected, db_name=args.db, out_dir=args.out_dir
    )
    for name, path in written.items():
        print(f"  {name:<11} {path}")
    return written


if __name__ == "__main__":
    main()
