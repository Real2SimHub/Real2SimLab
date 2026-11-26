# r2s-agent-gemini-cli: the twin runtime plus Gemini CLI, for building twins with it.
# The API key is passed at run time only (GEMINI_API_KEY), never baked in.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM node:22-bookworm-slim AS node

FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG GEMINI_CLI_VERSION=0.68.0
ARG NPM_REGISTRY=https://registry.npmjs.org
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources; fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --registry "${NPM_REGISTRY}" "@google/gemini-cli@${GEMINI_CLI_VERSION}" \
    && npm cache clean --force \
    && gemini --version
LABEL org.opencontainers.image.title="r2s-agent-gemini-cli"
