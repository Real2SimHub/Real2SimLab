#!/usr/bin/env bash
# Push the redistributable runtime images to one or more registries (used by the release
# workflow, which pushes to the owner's GHCR namespace; also usable by hand after `docker login`).
#
#   scripts/publish_images.sh --targets "ghcr.io/<owner>" r2s-python r2s-agent-codex
#
# Images are built first with `r2s env build`, under their local names ghcr.io/real2simhub/<name>:<tag>.
# r2s-agent-claude-code contains proprietary software and is refused.
set -euo pipefail

TAG="${IMAGE_TAG:-1.0}"
LOCAL_PREFIX="${R2S_IMAGE_PREFIX:-ghcr.io/real2simhub/}"
targets=""
images=()
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) targets="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) images+=("$1"); shift ;;
  esac
done
if [ -z "$targets" ] || [ "${#images[@]}" -eq 0 ]; then
  echo "usage: $0 --targets \"REGISTRY/NAMESPACE ...\" IMAGE..." >&2
  exit 2
fi
for image in "${images[@]}"; do
  if [ "$image" = "r2s-agent-claude-code" ]; then
    echo "refusing to publish $image: Claude Code is proprietary software" >&2
    exit 1
  fi
  source_ref="${LOCAL_PREFIX}${image}:${TAG}"
  docker image inspect "$source_ref" > /dev/null
  for target in $targets; do
    dest="${target%/}/${image}:${TAG}"
    echo "pushing $source_ref -> $dest"
    docker tag "$source_ref" "$dest"
    docker push "$dest"
  done
done
