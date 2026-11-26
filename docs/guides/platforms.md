# Platforms

Real2SimLab is developed and tested on Linux (x86-64, Ubuntu 24.04, Docker Engine and
bubblewrap, NVIDIA GPUs). This page says what to expect elsewhere.

| | Linux | macOS (Docker Desktop) | Windows |
|---|---|---|---|
| Status | tested (CI and development) | **not verified yet**; the CI checks job runs on macOS with the `local` backend | use **WSL 2** (Linux); native Windows is not supported |
| `docker` backend | yes | expected to work (see below) | inside WSL 2, with Docker Desktop's WSL integration or Docker Engine in WSL |
| `bwrap` backend | yes | no (Linux only) | inside WSL 2 |
| `local` backend | yes (no isolation) | yes (no isolation) | inside WSL 2 |
| GPU tasks (Blender renders, 4DCodeBench) | NVIDIA, with the Container Toolkit | no | NVIDIA through WSL 2's GPU support (not verified) |

## macOS with Docker Desktop

What should work: the synthetic suite, IRIS, agents in containers, Harbor export. Things to
know:

- **Shared folders.** Twins and agents see host directories through bind mounts. Docker Desktop
  shares `/Users`, `/Volumes`, `/private`, `/tmp` and `/var/folders` by default; keep tasks and
  run directories under one of them (or add the path in Docker Desktop's settings).
- **Apple Silicon.** `r2s-python` and the agent images build natively for arm64. The
  `r2s-blender-4.0` image downloads an x86-64 Blender and needs emulation
  (`--platform linux/amd64`, slow); 4DCodeBench's images are x86-64 with CUDA and do not run.
- **Resources.** `--cpus` and `--memory` limits apply inside Docker Desktop's virtual machine;
  give the VM enough of both in its settings.
- **No GPU** inside containers: GPU tasks (PhysCodeGym renders, 4DCodeBench) cannot run.

Please report what you find with `r2s doctor --json` attached (an issue template asks for it).

## Slow or blocked networks

Every download Real2SimLab makes can be pointed at a mirror:

| What | How |
|---|---|
| Runtime images | `R2S_IMAGE_MIRRORS=<registry>/<namespace>/[,...]`: tried before the project's registry (`ghcr.io/real2simhub/`) when an image is pulled (`r2s env pull`, or automatically) |
| Docker Hub base images (when building) | a registry mirror in Docker's `/etc/docker/daemon.json` (`"registry-mirrors"`) |
| Building images | `r2s env build NAME --pip-index-url ... --apt-mirror ... --npm-registry ... --network host` |
| Building behind a proxy | `r2s env build NAME --network host --build-arg HTTPS_PROXY --build-arg HTTP_PROXY` (a name alone takes the value from your shell; Docker keeps these out of the image's history) |
| Python packages | `pip install -i <PyPI mirror> -e .` in your clone (the dependencies come from PyPI) |
| Benchmark data on Hugging Face | `HF_ENDPOINT=<mirror>` (honoured by `huggingface_hub`) |
| Machines without Docker | `r2s env unpack r2s-python --from-url <mirror URL>` |

A failed `r2s env build` names the option that would have helped when it recognises a download
failure in the build output.

A request to Hugging Face that gets no answer ends after `HF_HUB_DOWNLOAD_TIMEOUT` seconds (10
unless set) and is retried a few times before the command reports a failed download; on a slow
connection, set it higher (for example `HF_HUB_DOWNLOAD_TIMEOUT=60`).

## Windows

Install WSL 2 with a Linux distribution, then follow the Linux instructions inside it. Real2SimLab
relies on POSIX user ids for its sandboxes (files written by a twin belong to you), which native
Windows Python does not provide.
