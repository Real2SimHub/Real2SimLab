# Run on a cluster without Docker

University clusters rarely allow Docker. The `bwrap` backend runs twins with
[bubblewrap](https://github.com/containers/bubblewrap) — an unprivileged sandbox most
distributions ship — on a root file system unpacked from the same images.

## 1. Check bubblewrap

```bash
r2s doctor
```

`backend bwrap: ready` means bwrap can start the sandbox every run gets: it creates user
namespaces and mounts a new `/proc` in them. On Ubuntu 24.04, AppArmor forbids user namespaces
by default (`bwrap cannot create namespaces`); an administrator allows them for bwrap with:

```bash
printf 'abi <abi/4.0>,\ninclude <tunables/global>\nprofile bwrap /usr/bin/bwrap flags=(unconfined) {\n  userns,\n}\n' \
  | sudo tee /etc/apparmor.d/bwrap && sudo apparmor_parser -r /etc/apparmor.d/bwrap
```

A container or Kubernetes pod that masks its own `/proc` does not let bwrap mount a new one
(`bwrap cannot mount a new /proc here`), and unpacking images does not change that: run on the
host, in a container whose `/proc` is not masked, or with Docker. On a machine without Docker,
`backend docker` is `ERR` and `r2s doctor` exits with status 1; here the bwrap lines are the ones
that matter (once bwrap is ready, `image r2s-python (bwrap)` says whether its image is unpacked).

## 2. Bring the images

The simplest way: download the runtime image published with this release (its SHA-256 is
checked before anything is unpacked):

```bash
r2s env unpack r2s-python --from-url
```

`--from-url URL` takes another location (a mirror, a shared drive served over HTTP) and
`--sha256` the expected hash when no `URL.sha256` file sits next to it (`--sha256` goes with
`--from-url` only). A download that gets no
answer is the network's or the proxy's (`--sha256` does not help there): the error names the
proxy variables this shell sets. A tarball already on this machine is not a URL: `--from-tar`
unpacks it.

Without access to the release, on a machine with Docker:

```bash
r2s env build r2s-python
r2s env export r2s-python --out r2s-python.tar
```

Copy the tarball to the cluster, then:

```bash
r2s env unpack r2s-python --from-tar r2s-python.tar
```

It unpacks into `~/.cache/real2simlab/rootfs/` (set `R2S_CACHE_DIR` to put it on a larger
disk). No root is needed: device nodes are skipped and set-id bits dropped. Each layer is
written as it is read, so unpacking needs little memory, whatever the size of the image. Each
layer is also checked against the digest the image lists for it, as `docker load` does: a copy
damaged on the way (one changed byte is enough) is refused, naming the layer, and nothing of it
is unpacked; copy the file again. The
tarball names the image it holds (`r2s env export` and `docker save` write its name into it):
it is unpacked only as that image, under any registry prefix (`R2S_IMAGE_PREFIX` where it was
exported), so a wrong file or a misspelt name is refused with what the file holds, instead of
becoming a runtime without what its tasks need. A tarball saved by image id names no image: it
is unpacked as the name given, with a note that this could not be checked. Where Docker is
available, `r2s env unpack r2s-python` does both steps. Unpacking an image again reuses what
is there (and says so); `--force` extracts it again, e.g. after a cleaner removed files from
the cache (without Docker, give `--from-tar` again: `--force` alone takes the image from
Docker). An image whose whole tree a cleaner removed (its record `image.json` left) is
extracted again by the same command, `--force` or not. An unpack killed outright (`kill -9`,
the out-of-memory killer) leaves its half-extracted copy beside the image; the next unpack on
the same machine removes it, once no process holds it (never one of another machine sharing
the disk, whose command may still run, nor one left before this machine restarted: remove those
yourself once no `r2s` command uses the cache). A run whose runtime cannot start its interpreter, or
lacks what every Lab runtime holds (NumPy, SciPy, the twin SDK, ffmpeg), is reported as an
infrastructure error with this fix, not as the program's failure; a run whose sandbox bwrap
itself could not set up is an infrastructure error that says so, with bwrap's fix (the image
is not at fault there).

## 3. Run

```bash
r2s eval --tasks tasks --method oracle --method static --backend bwrap --out runs/cluster
```

or `backend = "bwrap"` in an experiment file. Twins see the image as a read-only `/`, their
program read-only, an empty output directory and `/tmp`; there is no network and nothing else
of the host.

A command keeps to the unpacked image it found when it started: `r2s env unpack --force`
refuses to replace an image while a program runs on it, and an image replaced or removed anyway
(by hand, or with `--force` between two runs) makes that command's remaining runs
infrastructure errors, which running the command again redoes.

## GPUs

GPU twins work too: every `/dev/nvidia*` node is bound into the sandbox and a copy of the host's
NVIDIA user-space driver libraries (made once per driver version, in the cache) is mounted and
put on `LD_LIBRARY_PATH`. `r2s doctor --gpu` checks it; select GPUs with
`CUDA_VISIBLE_DEVICES` as usual. Two official scorers are the exception: 4DCodeBench's and
Video2World's evaluator run in Docker (with the NVIDIA Container Toolkit) whatever the backend,
so those two benchmarks are scored only where Docker is.

## Limits

| | bwrap | docker |
|---|---|---|
| time limit | yes | yes |
| CPU limit | by affinity, where `taskset` (util-linux) is installed: sandboxes that run at the same time on one machine, in one command or in several of yours, get CPUs of their own while there are enough (then they share the least used ones, with a warning); otherwise **no**, and records say so | yes (`--cpus`) |
| memory, process count | where a systemd user session delegates cgroup v2 controllers (most desktops; ask your admin on clusters), otherwise **no** — every run record lists what was enforced | yes |
| network allowlist for agents | **no**: agents need `network="open"` or the Docker backend | yes |

Agents on a cluster: `mini_swe` works as is (its harness on the login node talks to the model,
every command runs offline in bwrap); CLI agents need `network="open"`, which every report then
marks.

The bwrap backend probes once whether `systemd-run --user --scope` can limit `memory` and
`pids`; `r2s doctor` shows the result (`backend bwrap: ready; limits enforced: ...`), and each
run record's `limits_enforced` lists the limits that applied. Set `R2S_BWRAP_CGROUP=0` to run
without the scope. Where systemd takes it (253 and later), the scope also has
`OOMPolicy=continue`: a program that runs out of its memory is killed inside the sandbox, its
own failure, rather than the whole scope being stopped with bwrap in it.

Your r2s commands on one machine (two terminals, a sweep, the reward workers of a trainer)
agree on CPUs through lock files in `/tmp/r2s-cpus-<uid>`, one per CPU, which a command holds
while a sandbox of its own runs there; the system releases them when a command ends, even
with `kill -9`. Commands of other users, or in another `/tmp` (a container of their own), are
not counted: give each its own CPUs (a batch scheduler's allocation, `taskset -c`), or use
the Docker backend, whose `--cpus` shares the machine.

## Batch schedulers

Run one experiment per job; the run directory is the unit of resumption, so a job that hits
its time limit continues when resubmitted with the same command. Use `--concurrency` to match
the cores you requested: the sandboxes of a job are pinned to the CPUs the scheduler gave it.
