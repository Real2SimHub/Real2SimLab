# r2s-video2world-isaac: the Isaac Sim interpreter of Video2World's RoboDojo and in-house
# evaluators. It follows RoboDojo's own Dockerfile at the commit Video2World pins (25691aa):
# CUDA 12.8, Python 3.11, PyTorch 2.7.0 (cu128), Isaac Sim 5.1.0 from NVIDIA's package index,
# RoboDojo's IsaacLab and cuRobo forks built from its submodules, then RoboDojo's re-pins. The
# Python is a pinned python-build-standalone build in a venv (RoboDojo uses a conda env).
#
# Isaac Sim is NVIDIA software under the NVIDIA Isaac Sim / Omniverse license: the build refuses
# to run unless ACCEPT_EULA=Y, which `r2s env build` passes only when R2S_ACCEPT_NVIDIA_EULA=1.
# The image is built locally and never published. RoboDojo's assets (41 GB) are not in the image:
# the `video2world` scorer mounts them at /opt/RoboDojo/Assets. Video2World's code is mounted
# too; the image holds only the simulator stack and RoboDojo's code.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

FROM ${CUDA_IMAGE} AS source
ARG ROBODOJO_COMMIT=25691aa78fb34bbbe798aa3d880d57ef2788f696
ARG APT_MIRROR=
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g; s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list; \
        rm -f /etc/apt/sources.list.d/cuda*.list; fi \
    && apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# RoboDojo (MIT) with the three submodules its Dockerfile builds: IsaacLab (BSD-3-Clause) and
# cuRobo (Apache-2.0) forks, XPolicyLab (Apache-2.0), at the commits RoboDojo records.
RUN git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /opt/RoboDojo \
    && cd /opt/RoboDojo && git checkout --detach "${ROBODOJO_COMMIT}" \
    && git submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab \
    && git submodule status

FROM ${CUDA_IMAGE}
SHELL ["/bin/bash", "-c"]
ARG ACCEPT_EULA=
RUN if [ "${ACCEPT_EULA}" != "Y" ]; then \
        echo "Isaac Sim requires accepting NVIDIA's license: build with R2S_ACCEPT_NVIDIA_EULA=1" >&2; \
        exit 1; fi
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG ISAAC_INDEX_URL=https://pypi.nvidia.com
ARG PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.11.18%2B20261026-x86_64-unknown-linux-gnu-install_only.tar.gz
ARG PYTHON_SHA256=a87ee63646c8f8438b7af3954b2135ed2af91d033c631f12a8f20af903046d06
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDA_HOME=/usr/local/cuda \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
# RoboDojo's system packages: build tools, ffmpeg, and the headless OpenGL / EGL / Vulkan
# libraries Isaac Sim needs (libxt6 for its MaterialX render libraries).
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://archive.ubuntu.com|${APT_MIRROR}|g; s|http://security.ubuntu.com|${APT_MIRROR}|g" /etc/apt/sources.list; \
        rm -f /etc/apt/sources.list.d/cuda*.list; fi \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config git wget curl ca-certificates ffmpeg \
        libgl1 libglu1-mesa libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libice6 \
        libxrandr2 libxi6 libxcursor1 libxinerama1 libegl1 libvulkan1 vulkan-tools libxt6 \
    && rm -rf /var/lib/apt/lists/*
# Headless EGL through the NVIDIA driver the NVIDIA Container Toolkit injects at run time; one
# Vulkan ICD only (two make RTX device initialisation hang, as RoboDojo found).
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
# Python 3.11 (python-build-standalone, checked against its published sha256) and a venv.
RUN curl -fsSL "${PYTHON_URL}" -o /tmp/python.tar.gz \
    && echo "${PYTHON_SHA256}  /tmp/python.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/python && tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 \
    && rm /tmp/python.tar.gz \
    && /opt/python/bin/python3.11 -m venv /opt/robodojo
ENV VIRTUAL_ENV=/opt/robodojo \
    PATH=/opt/robodojo/bin:/usr/local/cuda/bin:${PATH}
# RoboDojo's setup_base_deps (scripts/requirements.txt without its development tools ruff and
# pre-commit), then setup_isaacsim.
RUN python -m pip install --index-url "${PIP_INDEX_URL}" --upgrade pip \
    && python -m pip install --index-url "${PIP_INDEX_URL}" \
        huggingface_hub transforms3d ffmpeg open3d msgpack-numpy \
    && python -m pip install --index-url "${PIP_INDEX_URL}" \
        opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn \
    && python -m pip install --index-url "${PIP_INDEX_URL}" numpy==1.26.0
RUN python -m pip install --index-url "${PIP_INDEX_URL}" \
        numpy==1.26.0 typing_extensions==4.12.2 filelock==3.13.1 \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" --extra-index-url "${PIP_INDEX_URL}" \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    && python -m pip install --index-url "${PIP_INDEX_URL}" --extra-index-url "${ISAAC_INDEX_URL}" \
        "isaacsim[all,extscache]==5.1.0"
COPY --from=source /opt/RoboDojo /opt/RoboDojo
WORKDIR /opt/RoboDojo
# setup_isaaclab, setup_curobo and repin_after_curobo; then the submodules' .git pointers go
# (cuRobo reads its version from git when one exists, which fails without the repository).
RUN cd third_party/IsaacLab && PIP_INDEX_URL="${PIP_INDEX_URL}" ./isaaclab.sh --install none
RUN cd third_party/curobo \
    && (python -m pip uninstall -y nvidia-curobo curobo || true) \
    && python -m pip install --index-url "${PIP_INDEX_URL}" -e ".[cu12]" --no-build-isolation
RUN python -m pip install --index-url "${PIP_INDEX_URL}" \
        numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1 websockets==12.0 \
        scipy==1.15.3 warp-lang==1.11.0 \
    && find /opt/RoboDojo -name .git -prune -exec rm -rf {} + ; true
# What Video2World's own code needs in this interpreter (its pyproject's dependencies; trimesh at
# the version of its MuJoCo extra, which supports NumPy 1.26).
RUN python -m pip install --index-url "${PIP_INDEX_URL}" \
        trimesh==4.12.2 rtree embreex==4.4.0 "imageio[ffmpeg]>=2.31" numpy==1.26.0 \
    && python -c "import isaacsim, torch, trimesh, scipy, cv2; print('torch', torch.__version__)"
# The scorer runs as the host user: Isaac Sim writes caches inside its package.
RUN mkdir -p /opt/RoboDojo/Assets \
    && chmod -R a+rwX /opt/robodojo/lib/python3.11/site-packages/isaacsim /opt/RoboDojo
ENV ASSETS_PATH=/opt/RoboDojo \
    V2W_PYTHON=/opt/robodojo/bin/python \
    V2W_ISAAC_PYTHON=/opt/robodojo/bin/python \
    V2W_ROBODOJO_SOURCE=/opt/RoboDojo \
    V2W_FFMPEG=/usr/bin/ffmpeg
LABEL org.opencontainers.image.title="r2s-video2world-isaac"
