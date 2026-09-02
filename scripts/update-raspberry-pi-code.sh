#!/usr/bin/env bash

# Update application code on an already provisioned BearVision Edge device.
# Dependency manifests must match the installed deployment; otherwise use the
# full setup-raspberry-pi.sh flow.

set -Eeuo pipefail

readonly SERVICE_NAME="bearvision-edge-control"
readonly INSTALL_DIR="/opt/bearvision"
readonly STATE_DIR="/var/lib/bearvision"
readonly SERVICE_USER="bearvision"

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

log() {
    printf '[BearVision code update] %s\n' "$*"
}

die() {
    printf '[BearVision code update] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '[BearVision code update] ERROR: command failed at line %s (exit %s)\n' \
        "${BASH_LINENO[0]}" "$exit_code" >&2
    exit "$exit_code"
}

trap on_error ERR

[[ $EUID -eq 0 ]] || die "run this script as root (for example, with sudo)"
id "$SERVICE_USER" >/dev/null 2>&1 || die "run the full setup first (missing user $SERVICE_USER)"
[[ -d $INSTALL_DIR && -d $INSTALL_DIR/.venv ]] || \
    die "run the full setup first (missing $INSTALL_DIR runtime)"
systemctl cat "$SERVICE_NAME.service" >/dev/null 2>&1 || \
    die "run the full setup first (missing $SERVICE_NAME.service)"

for relative_path in \
    pyproject.toml \
    uv.lock \
    apps/edge-control/package.json \
    apps/edge-control/pnpm-lock.yaml; do
    [[ -f "$SOURCE_DIR/$relative_path" ]] || die "deployment is missing $relative_path"
    [[ -f "$INSTALL_DIR/$relative_path" ]] || \
        die "installed runtime is missing $relative_path; run the full setup"
    cmp --silent "$SOURCE_DIR/$relative_path" "$INSTALL_DIR/$relative_path" || \
        die "$relative_path changed; run a full deployment to update dependencies"
done

for required_directory in src apps/edge-control specs/scenarios; do
    [[ -d "$SOURCE_DIR/$required_directory" ]] || \
        die "deployment is missing $required_directory"
done

SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

log "Synchronizing application code"
rsync --archive --delete "$SOURCE_DIR/src/" "$INSTALL_DIR/src/"
rsync --archive --delete \
    --exclude node_modules \
    --exclude dist \
    "$SOURCE_DIR/apps/edge-control/" "$INSTALL_DIR/apps/edge-control/"
rsync --archive --delete \
    "$SOURCE_DIR/specs/scenarios/" "$INSTALL_DIR/specs/scenarios/"
chown -R "$SERVICE_USER:$SERVICE_GROUP" \
    "$INSTALL_DIR/src" \
    "$INSTALL_DIR/apps/edge-control" \
    "$INSTALL_DIR/specs/scenarios"

log "Building Edge Control with installed dependencies"
runuser -u "$SERVICE_USER" -- env \
    HOME="$STATE_DIR" \
    XDG_CACHE_HOME="$STATE_DIR/cache" \
    pnpm --dir "$INSTALL_DIR/apps/edge-control" build
runuser -u "$SERVICE_USER" -- \
    node --check "$INSTALL_DIR/apps/edge-control/server/server.mjs"

log "Restarting $SERVICE_NAME"
systemctl restart "$SERVICE_NAME.service"
systemctl is-active --quiet "$SERVICE_NAME.service" || \
    die "$SERVICE_NAME did not start; inspect journalctl -u $SERVICE_NAME -n 100"

log "Code update complete"
printf 'Control UI: http://%s:4310\n' "$(hostname -s)"
