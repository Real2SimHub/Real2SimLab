# r2s-agent-claude-code: the twin runtime plus Claude Code, for building twins with it.
# Credentials are passed at run time only, never baked in.
#
# Claude Code is proprietary software ("All rights reserved"; see its LICENSE.md and
# https://code.claude.com/docs/en/legal-and-compliance). Build this image locally; do not
# publish it. The build prints the package's license note.
ARG BASE_IMAGE=ghcr.io/real2simhub/r2s-python:1.0
FROM node:22-bookworm-slim AS node

FROM ${BASE_IMAGE}
ARG APT_MIRROR=
ARG CLAUDE_CODE_VERSION=2.1.323
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
    && npm install -g --registry "${NPM_REGISTRY}" "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force \
    && cat "$(npm root -g)/@anthropic-ai/claude-code/LICENSE.md" \
    && claude --version
LABEL org.opencontainers.image.title="r2s-agent-claude-code" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary-Anthropic" \
      org.opencontainers.image.description="Local build only: contains proprietary software"
