"""Export a run directory's db to extended XYZ, for viewing in OVITO or ASE's GUI.

    python scripts/export_xyz.py RUN_DIR                      # simulation.db -> simulation.xyz
    python scripts/export_xyz.py RUN_DIR --db equilibration.db

Thin CLI over :func:`asmcmc.mc.diagnostics.export_xyz`.
"""

import argparse

from asmcmc.mc.diagnostics import export_xyz


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="Run directory holding the db.")
    parser.add_argument(
        "--db", default="simulation.db", help="Which db to export (default: simulation.db)."
    )
    args = parser.parse_args(argv)
    print(export_xyz(args.run_dir, args.db))


if __name__ == "__main__":
    main()
