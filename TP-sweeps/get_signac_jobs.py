import signac

project = signac.get_project()

# Running across all beads and all 
# job_filter = {"P":1}   # Filters jobs that match this criteria
job_filter = {} # No filter
jobs = project.find_jobs(job_filter)   # Finds eligible jobs.
with open("workspace.txt", "w") as f:
    for job in jobs:
        if not job.isfile("equilibriation.db"):    # Filter for jobs that haven't run equilibriation
            print(job.id, file=f)

