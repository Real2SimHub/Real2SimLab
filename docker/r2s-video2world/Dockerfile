# r2s-video2world: Video2World's ManiSkill / SAPIEN and MuJoCo evaluation environment. First the
# `sim` extra of its pyproject at the versions its reference results were produced with (torch,
# unpinned there, is pinned here, CPU build: physics runs on PhysX CPU). SAPIEN still needs a Vulkan device to load
# meshes: the NVIDIA GPU, through the NVIDIA Container Toolkit (graphics capability, set below);
# Debian 12's Mesa lavapipe (installed) is too old for SAPIEN 3.0.3. Used by
# the `video2world` scorer and by agents that validate and replay their packages. The benchmark's
# code is not in the image: the scorer mounts the pinned checkout, and agents get only the
# video2sim library in their workspace.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.15.0
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers libegl1 libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" --extra-index-url "${PIP_INDEX_URL}" \
        "torch==${TORCH_VERSION}+cpu" \
    && pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        mani_skill==3.0.1 sapien==3.0.3 pytorch_kinematics==0.7.6 numpy==2.4.6 scipy==1.17.1 \
        trimesh==5.0.0 rtree embreex==4.4.0 opencv-python-headless "imageio[ffmpeg]>=2.31" \
    && python -c "import sapien, mani_skill, torch; print('sapien', sapien.__version__, 'mani_skill', mani_skill.__version__, 'torch', torch.__version__)"
# ManiSkill assets the evaluator loads, downloaded here because evaluation runs offline; ManiSkill
# checks each archive against the sha256 in its registry:
# - xarm6: the Robotiq 2F-85 meshes of the DROID rig (ManiSkill-XArm6 v0.1.1, Apache-2.0);
# - bridge_v2_real2sim: the Bridge real2sim scene assets of the `fb` evaluator's environment
#   (Hugging Face haosulab/ManiSkill_bridge_v2_real2sim, MIT; sha256 618512a2...).
ENV MS_ASSET_DIR=/opt/maniskill/data MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
RUN python -m mani_skill.utils.download_asset xarm6 -y \
    && python -m mani_skill.utils.download_asset bridge_v2_real2sim -y \
    && chmod -R a+rX /opt/maniskill && ls /opt/maniskill/data/data/*
# - widowx250s: the nominal robot the `fb` evaluator builds the scene with (ManiSkill-WidowX250S
#   v0.2.0, from SimplerEnv). No license is declared for it: fetched from its authors' release
#   when the image is built, and this image is never published (D-043).
RUN python -m mani_skill.utils.download_asset widowx250s -y \
    && chmod -R a+rX /opt/maniskill && ls /opt/maniskill/data/data/robots
# The reconstructed-twin and cloth evaluators (MuJoCo) run in a second interpreter, the `twin`
# extra of Video2World's pyproject at its pinned versions (it needs NumPy 1.x). The scorer and
# agents reach it as V2W_TWIN_PYTHON.
RUN python -m venv /opt/v2w-twin \
    && /opt/v2w-twin/bin/pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" \
        mujoco==3.11.0 numpy==1.26.4 scipy==1.17.1 trimesh==4.12.2 "imageio[ffmpeg]>=2.31" \
        "opencv-python-headless<4.12" \
    && /opt/v2w-twin/bin/python -c "import mujoco, numpy, imageio_ffmpeg; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
ENV V2W_TWIN_PYTHON=/opt/v2w-twin/bin/python
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
LABEL org.opencontainers.image.title="r2s-video2world"
