# r2s-agent-codex: the twin runtime plus the Codex CLI, for building twins with Codex.
# Credentials are never baked in: they are copied into a temporary CODEX_HOME at run time.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM node:22-bookworm-slim AS node

FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG CODEX_VERSION=0.167.2
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
    && npm install -g --registry "${NPM_REGISTRY}" "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && codex --version
LABEL org.opencontainers.image.title="r2s-agent-codex"
