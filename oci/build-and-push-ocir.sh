#!/usr/bin/env bash
set -euo pipefail

: "${OCI_REGION:?Set OCI_REGION, for example us-phoenix-1}"
: "${OCI_TENANCY_NAMESPACE:?Set OCI_TENANCY_NAMESPACE}"
: "${OCIR_REPOSITORY:?Set OCIR_REPOSITORY}"
: "${OCIR_AUTH_TOKEN:?Set OCIR_AUTH_TOKEN without committing it}"

command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }

registry="${OCI_REGION}.ocir.io"
image="${registry}/${OCI_TENANCY_NAMESPACE}/${OCIR_REPOSITORY}:visionid-$(git rev-parse --short HEAD)"

echo "$OCIR_AUTH_TOKEN" | docker login "$registry" \
  --username "${OCI_TENANCY_NAMESPACE}/<OCI_USERNAME>" \
  --password-stdin

docker build -t "$image" .
docker push "$image"
printf 'OCIR_IMAGE_URL=%s\n' "$image"
