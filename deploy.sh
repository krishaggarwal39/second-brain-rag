#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Second Brain RAG - production deployment
#
#   ./deploy.sh                      validate .env, build, start, wait for readiness
#   ./deploy.sh --generate-secrets   also fill missing secrets in .env (random hex)
#
# Secret VALUES are never printed; only key names are reported.
# -----------------------------------------------------------------------------
set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
READY_URL="${READY_URL:-http://localhost:${FRONTEND_HOST_PORT:-80}/api/health/ready}"
READY_TIMEOUT="${READY_TIMEOUT:-300}"
GENERATE=0

for arg in "$@"; do
    case "$arg" in
        --generate-secrets) GENERATE=1 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Secrets compose refuses to start without, plus the LLM key the app needs.
SECRET_KEYS=(JWT_SECRET API_KEY CHROMA_AUTH_TOKEN REDIS_PASSWORD NEO4J_PASSWORD POSTGRES_PASSWORD)
REQUIRED_KEYS=("${SECRET_KEYS[@]}" GOOGLE_API_KEY)

echo "==> Second Brain RAG production deployment"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is not installed." >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose v2 is required." >&2; exit 1; }

if [ ! -f "$ENV_FILE" ]; then
    if [ -f .env.example ]; then
        cp .env.example "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        echo "Created $ENV_FILE from .env.example. Set GOOGLE_API_KEY (and optionally GROQ_API_KEY),"
        echo "then run: ./deploy.sh --generate-secrets"
        exit 1
    fi
    echo "ERROR: neither .env nor .env.example exists." >&2
    exit 1
fi

# Value of KEY in .env (empty when missing). Not echoed anywhere.
env_value() {
    local line
    line=$(grep -E "^$1=" "$ENV_FILE" | tail -n 1 || true)
    line="${line#*=}"
    line="${line%\"}"; line="${line#\"}"
    printf '%s' "$line"
}

is_placeholder() {
    case "$1" in
        ""|*change-me*|*CHANGE_ME*|*your-*|*default-*|*dev-only-*) return 0 ;;
        *) return 1 ;;
    esac
}

set_env_value() {
    local key="$1" value="$2" tmp
    tmp=$(mktemp)
    grep -vE "^${key}=" "$ENV_FILE" > "$tmp" || true
    printf '%s=%s\n' "$key" "$value" >> "$tmp"
    mv "$tmp" "$ENV_FILE"
}

if [ "$GENERATE" -eq 1 ]; then
    command -v openssl >/dev/null 2>&1 || { echo "ERROR: openssl is required for --generate-secrets." >&2; exit 1; }
    for key in "${SECRET_KEYS[@]}"; do
        if is_placeholder "$(env_value "$key")"; then
            set_env_value "$key" "$(openssl rand -hex 32)"
            echo "    generated $key"
        fi
    done
fi

chmod 600 "$ENV_FILE"

missing=()
for key in "${REQUIRED_KEYS[@]}"; do
    if is_placeholder "$(env_value "$key")"; then
        missing+=("$key")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: these keys are missing or still placeholders in $ENV_FILE:" >&2
    printf '    %s\n' "${missing[@]}" >&2
    echo "Fill them in, or run ./deploy.sh --generate-secrets for the generated ones." >&2
    exit 1
fi
echo "    .env has all required keys (values not shown)"

echo "==> Validating compose configuration"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config -q

echo "==> Building and starting containers"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build

echo "==> Waiting for readiness at $READY_URL (timeout ${READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
until curl -fsS "$READY_URL" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "ERROR: the stack did not become ready in time." >&2
        echo "Check: docker compose -f $COMPOSE_FILE ps   and   docker compose -f $COMPOSE_FILE logs backend" >&2
        exit 1
    fi
    sleep 5
done

echo "==> Deployment ready."
echo "    Logs: docker compose -f $COMPOSE_FILE logs -f"
echo "    Note: plain HTTP must not be exposed publicly. See README \"HTTPS\" for the tls profile."
