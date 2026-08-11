# Incorporating signac-flow for job management

I just copied over a large portion of `scripts/single_state_run_1atm_300K.py` and replaced hardcoded parameters with `job.sp["param"]`, where the parameters are defined and initialized in `./init.py`

I've gitignored signac-related files, so you will need to run the following and populate (either locally or on chtc) the project results:
* `signac init` initializes a signac project.
* `python init.py` initializes the workspace for the parameter sweeps. You can change `init.py` to include whatever parameters you want.
  
Additional useful signac commands for local running:
* `signac view` - creates symlinks of the `workspace/hashes` into interpretable folder structures.
* `signac find key_1 value_1 ... key_n value_n` - returns a list of hashes that matches jobs that match the listed key value pairs. Specific example:
  * `signac find T 250` - returns a list of hashes that matches jobs that have T=250K (3 jobs, since there are 3 corresponding pressures)
  * `signac find T 250 P 1.75` - returns a list of hashes that matches jobs that have T=250K, P=1.75atm
* `python project.py status` - views the status of the jobs.
* `python project.py run` - runs the pipeline `project.py` for all eligible jobs.
* `python project.py run -f T 250 P 1.75` - filters for jobs matching T=250K and runs the pipeline defined in `project.py`.
  * Follows same syntax as `signac find` after the `-f` option 

Note: an eligible job is one whose preconditions are met and whose postconditions are not met.
* preconditions, postconditions defined by the labels in `project.py`


# For CHTC:
* tbd.