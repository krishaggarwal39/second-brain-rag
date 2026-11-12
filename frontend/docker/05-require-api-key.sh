#!/bin/sh
# Fail fast instead of proxying every request with an empty X-API-Key (silent 401/403s).
set -e
if [ -z "${API_KEY:-}" ]; then
    echo "ERROR: API_KEY is not set. nginx injects it into every /api request; refusing to start." >&2
    exit 1
fi
