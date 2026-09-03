#!/usr/bin/env bash

# Update application code on an already provisioned BearVision Edge device.
# Dependency manifests must match the installed deployment; otherwise use the
# full setup-raspberry-pi.sh flow.

set -Eeuo pipefail

readonly SERVICE_NAME="bearvision-edge-control"
readonly INSTALL_DIR="/opt/bearvision"
readonly SERVICE_USER="bearvision"
readonly SYSTEMCTL="/usr/bin/systemctl"
readonly RSYNC_PERMISSIONS="Du=rwx,Dg=rx,Dg+s,Do=,Fu=rw,Fg=r,Fo="

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

same_manifest_content() {
    cmp --silent <(tr -d '\r' < "$1") <(tr -d '\r' < "$2")
}

[[ $EUID -ne 0 ]] || die "run this script as the deployment user, not as root"
id "$SERVICE_USER" >/dev/null 2>&1 || die "run the full setup first (missing user $SERVICE_USER)"
[[ -d $INSTALL_DIR && -d $INSTALL_DIR/.venv ]] || \
    die "run the full setup first (missing $INSTALL_DIR runtime)"
"$SYSTEMCTL" cat "$SERVICE_NAME.service" >/dev/null 2>&1 || \
    die "run the full setup first (missing $SERVICE_NAME.service)"

for relative_path in \
    pyproject.toml \
    uv.lock \
    apps/edge-control/package.json \
    apps/edge-control/pnpm-lock.yaml; do
    [[ -f "$SOURCE_DIR/$relative_path" ]] || die "deployment is missing $relative_path"
    [[ -f "$INSTALL_DIR/$relative_path" ]] || \
        die "installed runtime is missing $relative_path; run the full setup"
    same_manifest_content "$SOURCE_DIR/$relative_path" "$INSTALL_DIR/$relative_path" || \
        die "$relative_path changed; run a full deployment to update dependencies"
done

for required_directory in src apps/edge-control specs/scenarios; do
    [[ -d "$SOURCE_DIR/$required_directory" ]] || \
        die "deployment is missing $required_directory"
done

for writable_directory in \
    "$INSTALL_DIR/src" \
    "$INSTALL_DIR/apps/edge-control" \
    "$INSTALL_DIR/specs/scenarios"; do
    [[ -w $writable_directory ]] || \
        die "$writable_directory is not writable; rerun full setup with --deploy-user $(id -un)"
done

log "Synchronizing application code"
rsync --archive --no-owner --no-group --chmod="$RSYNC_PERMISSIONS" --delete \
    "$SOURCE_DIR/src/" "$INSTALL_DIR/src/"
rsync --archive --no-owner --no-group --chmod="$RSYNC_PERMISSIONS" --delete \
    --exclude node_modules \
    --exclude dist \
    "$SOURCE_DIR/apps/edge-control/" "$INSTALL_DIR/apps/edge-control/"
rsync --archive --no-owner --no-group --chmod="$RSYNC_PERMISSIONS" --delete \
    "$SOURCE_DIR/specs/scenarios/" "$INSTALL_DIR/specs/scenarios/"

log "Building Edge Control with installed dependencies"
pnpm --dir "$INSTALL_DIR/apps/edge-control" build
node --check "$INSTALL_DIR/apps/edge-control/server/server.mjs"

log "Restarting $SERVICE_NAME"
if ! sudo -n /usr/bin/systemctl restart "$SERVICE_NAME.service"; then
    die "passwordless service restart is not configured; rerun the full setup with --deploy-user $(id -un)"
fi
"$SYSTEMCTL" is-active --quiet "$SERVICE_NAME.service" || \
    die "$SERVICE_NAME did not start; inspect journalctl -u $SERVICE_NAME -n 100"

log "Code update complete"
printf 'Control UI: http://%s:4310\n' "$(hostname -s)"
