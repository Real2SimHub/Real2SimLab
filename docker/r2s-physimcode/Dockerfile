# r2s-physimcode: the runtime of PhySimCode's 2D (SciPy) programs: r2s-python plus Matplotlib,
# whose FuncAnimation and FFMpegWriter PhySimCode's programs use to render their videos.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM ${BASE_IMAGE}
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" matplotlib==3.11.2 \
    && python -c "import matplotlib; matplotlib.use('Agg'); print('matplotlib', matplotlib.__version__)"
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
LABEL org.opencontainers.image.title="r2s-physimcode"
