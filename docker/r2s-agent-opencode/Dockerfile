# r2s-agent-opencode: the twin runtime plus OpenCode 1.19.5.
# API keys are passed at run time only and are never baked in.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM node:22-bookworm-slim AS node

FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG OPENCODE_VERSION=1.19.5
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
    && npm install -g --registry "${NPM_REGISTRY}" "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force \
    && opencode --version
LABEL org.opencontainers.image.title="r2s-agent-opencode"
