#!/usr/bin/env bash
set -euo pipefail

: "${OCI_COMPARTMENT_OCID:?Set OCI_COMPARTMENT_OCID}"
: "${OCI_AVAILABILITY_DOMAIN:?Set OCI_AVAILABILITY_DOMAIN}"
: "${OCI_PUBLIC_SUBNET_OCID:?Set OCI_PUBLIC_SUBNET_OCID}"
: "${OCIR_IMAGE_URL:?Set OCIR_IMAGE_URL}"

command -v oci >/dev/null || { echo 'OCI CLI is required.' >&2; exit 1; }
command -v jq >/dev/null || { echo 'jq is required.' >&2; exit 1; }

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

sed \
  -e "s#<OCI_COMPARTMENT_OCID>#$OCI_COMPARTMENT_OCID#g" \
  -e "s#<OCI_AVAILABILITY_DOMAIN>#$OCI_AVAILABILITY_DOMAIN#g" \
  -e "s#<OCI_PUBLIC_SUBNET_OCID>#$OCI_PUBLIC_SUBNET_OCID#g" \
  -e "s#<OCIR_IMAGE_URL>#$OCIR_IMAGE_URL#g" \
  "$(dirname "$0")/container-instance.example.json" > "$manifest"

oci container-instances container-instance create \
  --from-json "file://$manifest"
