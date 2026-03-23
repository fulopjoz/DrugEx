# LICH Compute: The Basics

> **Saved locally from https://github.com/lich-uct/lich-compute (Mar 20, 2026)**
> See ERRATA at the bottom for known discrepancies between documentation and actual cluster configuration.

In this guide, you will find some basic information about how to use our HPC cluster called `lich-compute`. This readme file contains instructions on how to run jobs and organize your work. Please, also pay attention to the [Dos & Don'ts](#dos--donts) section to make sure everyone has the best experience. Take the time to read the whole file at least on the surface level. It will help you deal with the most common issues encountered by the people before you :) If you have any suggestions, questions or problems, do not hesitate to contact us via the [issue tracker](https://github.com/lich-uct/lich-compute/issues). We are still working on this guide as well so any contributions in the form of [pull requests](./pulls) or other means are very appreciated.

## Infrastructure

The *central node* (`lich-compute`) is the machine you will login to and submit jobs to PBS Pro job scheduler on. Each submitted job is distributed to one of the available *worker nodes*.

### Hardware

- **lich-compute** -- head node for user access and data storage (HDD-EXT)
    - specs:
        - CPU AMD Epyc Milan 7313 (16-core, 3.0GHz)
        - 1 TB RAM
    - storage:
        - `/home` -- 915 TB persistent redundant storage (JBOD 60x 22 TB HDD SAS)

- **lich-compute-emgpu** -- GPU worker node (GPU-MDRN)
    - specs:
        - 2x CPU Intel Xeon Gold 6526Y (16-core, 2,8GHz) = 32 cores
        - 1 TB RAM
        - 8x GPU **Nvidia L40S**, 48 GB
    - storage:
        - `/scratch` -- 15 TB NVMe SSD scratch for local jobs (volatile)

- **lich-compute-emcpu-N** -- **9x** -- identical CPU worker nodes
    - specs:
        - 2x CPU AMD Epyc Genoa 9654 (96-core, 2,4GHz) = 192 cores
        - 2.2 TB RAM
    - storage:
        - `/scratch` -- 22.8 TB (6x 3.8 TB NVMe SSD) for local jobs (volatile)

- **lich-compute-in-N** -- **14x** -- the old GPU worker nodes from INMODOS
    - specs:
        - 1x24-core (48 threads) CPU (AMD Epyc 7401P 2.0GHz)
        - 64 GB RAM
        - up to 3 GPUs NVIDIA 2080 Ti (past guarantee, no longer being replaced)
    - storage:
        - `/scratch` -- 2 TB for local jobs (volatile)

### Queue Selection

- `all` (default) - catch-all queue
- `cpu` - CPU nodes, no GPU
- `gpu` - any CUDA-capable GPU
- `gpu-2080` - nodes with 2080Ti GPU
- `gpu-140` - nodes with L40S GPU

### Example Job Script

```bash
#!/bin/bash
#PBS -N basics
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=1:00:00
#PBS -m ae

DATADIR=/home/$USER/my_project/
SCRATCHDIR=/scratch/$USER/

echo "$PBS_JOBID is running on node `hostname -f` in $SCRATCHDIR" >> $DATADIR/jobs_info.txt

cp $DATADIR/example_input.txt $SCRATCHDIR || { echo >&2 "Error copying!"; exit 2; }
cd $SCRATCHDIR

# run computation
cat example_input.txt > output.txt

# copy results back
mkdir -p $DATADIR/$PBS_JOBID
cp output.txt $DATADIR/$PBS_JOBID || { echo >&2 "Copy failed! Files at `hostname -f`:`pwd`"; exit 4; }

# clean scratch
rm example_input.txt output.txt
```

### Dos & Don'ts

1. Do not use the central node for intensive calculations — use qsub.
2. Clean `/scratch/$USER` on worker nodes when job finishes or fails.
3. Only request resources the job actually needs (ncpus, ngpus, mem, walltime).
4. Don't manipulate huge amounts of data in/out of the network.

### Miscellaneous Tips

- Only use `/home/$USER/` for persistent storage.
- Copy large files to `/scratch/$USER/` before computation (SSD-backed, faster than /home HDD).
- Don't remove `~/.ssh/authorized_keys` (needed for NFS home mount).

---

## ERRATA: Known Discrepancies (verified Mar 20, 2026)

### `/scratch` is NOT local NVMe on worker nodes

The guide describes `/scratch` as local NVMe storage on each worker node. **This is incorrect as of the current cluster configuration.**

**Actual mount points on worker nodes (verified via SSH to emcpu-4, emcpu-5):**

| Path | Filesystem | Type | Notes |
|------|-----------|------|-------|
| `/home/` | `lich-compute:/home` | NFS4 | HDD pool, 859T, persistent |
| `/scratch/` | `lich-compute:/scratch` | NFS4 | **SSD pool, 11T, shared via NFS** |
| `/scratch-local/` | `/dev/md0` | XFS (local) | **Actual local NVMe RAID, 21T, but root-only (0755)** |

**Evidence:**
```
$ ssh lich-compute-emcpu-5 "df -hT /scratch/ /scratch-local/"
lich-compute:/scratch  nfs4   11T  417G  9.7T   5% /scratch       ← NFS!
/dev/md0               xfs    21T  150G   21T   1% /scratch-local  ← local NVMe
```

**Benchmark (100MB sequential write on emcpu-5):**
- `/scratch/` (NFS SSD): 670 MB/s
- `/home/` (NFS HDD): not tested, expected slower
- `/scratch-local/` (local NVMe): not testable (root-only permissions)

**Impact:** Using `/scratch/$USER/` as recommended in the guide still works and IS faster than `/home/` (SSD vs HDD backend), but it is NOT local I/O — it goes through NFS. For small-file-heavy workloads (thousands of tiny files), NFS metadata overhead dominates.

**Recommendation:** Open an issue at https://github.com/lich-uct/lich-compute/issues to either:
1. Set `/scratch-local/` permissions to 1777 (like `/tmp`) so users can use local NVMe
2. Or update the guide to accurately describe `/scratch` as shared NFS SSD
