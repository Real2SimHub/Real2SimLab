# r2s-blender-4.0: Blender 4.0.2 on top of the twin runtime, for Blender-based tasks
# (PhysCodeGym). Headless rendering: Cycles on CPU, or EEVEE/Cycles on a GPU through EGL
# (run with --gpus; NVIDIA_DRIVER_CAPABILITIES includes graphics).
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG BLENDER_URL=https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz
ARG BLENDER_SHA256=5583a5588736da8858c522ef17fff5d73be59c47a6fe91ad29c6f3263e22086a
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
       libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libxrandr2 \
       libxinerama1 libxcursor1 libsm6 libice6 libdbus-1-3 \
       libgl1 libegl1 libopengl0 libglx-mesa0 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "${BLENDER_URL}" -o /tmp/blender.tar.xz \
    && echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/blender \
    && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && blender --version
COPY blender_scene.py /opt/r2s/blender_scene.py
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    BLENDER_BIN=/opt/blender/blender
LABEL org.opencontainers.image.title="r2s-blender-4.0"
