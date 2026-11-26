# r2s-verifier: the twin runtime plus Real2SimLab's own dependencies, for verifiers of tasks
# exported to Harbor (the exported task copies Real2SimLab itself into the image).
# git: verifiers of 4DCodeBench tasks fetch its checker at a pinned commit when they are built.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
USER root
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" pydantic==2.14.3 tomli-w==1.2.0
LABEL org.opencontainers.image.title="r2s-verifier"
