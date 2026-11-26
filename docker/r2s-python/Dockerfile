# r2s-python: the CPU runtime for twins. Python 3.11, NumPy, SciPy, ffmpeg, the twin SDK.
# Built by `r2s env build r2s-python`; the build context also holds r2s_twin.py.
FROM python:3.11-slim-bookworm

ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" numpy==2.2.6 scipy==1.15.3

COPY r2s_twin.py /opt/r2s/sdk/r2s_twin.py
ENV PYTHONPATH=/opt/r2s/sdk \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp \
    HOME=/tmp
LABEL org.opencontainers.image.title="r2s-python" \
      org.opencontainers.image.description="Real2SimLab CPU runtime for twins" \
      org.opencontainers.image.licenses="Apache-2.0"
