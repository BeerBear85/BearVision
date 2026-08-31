#!/usr/bin/env bash

# Provision a 64-bit Raspberry Pi OS host as a BearVision edge device.
# Run from a BearVision checkout with:
#   sudo bash scripts/setup-raspberry-pi.sh

set -Eeuo pipefail

readonly SERVICE_NAME="bearvision-edge"
readonly DEFAULT_INSTALL_DIR="/opt/bearvision"
readonly DEFAULT_STATE_DIR="/var/lib/bearvision"
readonly DEFAULT_CONFIG_DIR="/etc/bearvision"
readonly DEFAULT_SERVICE_USER="bearvision"
readonly DEFAULT_UV_VERSION="0.12.7"

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
INSTALL_DIR="$DEFAULT_INSTALL_DIR"
STATE_DIR="$DEFAULT_STATE_DIR"
CONFIG_DIR="$DEFAULT_CONFIG_DIR"
SERVICE_USER="$DEFAULT_SERVICE_USER"
UV_VERSION="$DEFAULT_UV_VERSION"
DEVICE_ID=""
DEVICE_ID_WAS_SET=false
START_SERVICE=false

usage() {
    cat <<'EOF'
Usage: sudo bash scripts/setup-raspberry-pi.sh [options]

Set up BearVision as a systemd-managed edge service on 64-bit Raspberry Pi OS.

Options:
  --device-id ID       Edge identifier written to a newly created config file.
                       Defaults to the Pi hostname.
  --install-dir PATH   Application directory (default: /opt/bearvision).
  --state-dir PATH     Capture and scratch directory (default: /var/lib/bearvision).
  --service-user USER  Unprivileged runtime account (default: bearvision).
  --uv-version VERSION Pinned uv installer version (default: 0.12.7).
  --start              Start or restart the service after installation.
  -h, --help           Show this help.

The script preserves existing files in /etc/bearvision. Put Box credentials in
/etc/bearvision/bearvision.env; never add credentials to the repository.
EOF
}

log() {
    printf '[BearVision setup] %s\n' "$*"
}

warn() {
    printf '[BearVision setup] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[BearVision setup] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '[BearVision setup] ERROR: command failed at line %s (exit %s)\n' \
        "${BASH_LINENO[0]}" "$exit_code" >&2
    exit "$exit_code"
}

trap on_error ERR

require_value() {
    local option=$1
    local value=${2-}
    [[ -n "$value" ]] || die "$option requires a value"
}

while (($# > 0)); do
    case "$1" in
        --device-id)
            require_value "$1" "${2-}"
            DEVICE_ID=$2
            DEVICE_ID_WAS_SET=true
            shift 2
            ;;
        --install-dir)
            require_value "$1" "${2-}"
            INSTALL_DIR=$2
            shift 2
            ;;
        --state-dir)
            require_value "$1" "${2-}"
            STATE_DIR=$2
            shift 2
            ;;
        --service-user)
            require_value "$1" "${2-}"
            SERVICE_USER=$2
            shift 2
            ;;
        --uv-version)
            require_value "$1" "${2-}"
            UV_VERSION=$2
            shift 2
            ;;
        --start)
            START_SERVICE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ $EUID -eq 0 ]] || die "run this script as root (for example, with sudo)"
[[ -r /etc/os-release ]] || die "cannot identify the operating system"

# shellcheck source=/dev/null
source /etc/os-release
case "${ID:-} ${ID_LIKE:-}" in
    *debian*) ;;
    *) die "Raspberry Pi OS or another Debian-based OS is required" ;;
esac

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) die "a 64-bit ARM OS is required; install 64-bit Raspberry Pi OS" ;;
esac

[[ -r /proc/device-tree/model ]] || die "this host does not appear to be a Raspberry Pi"
PI_MODEL="$(tr -d '\000' < /proc/device-tree/model)"
[[ $PI_MODEL == *"Raspberry Pi"* ]] || die "this host does not appear to be a Raspberry Pi"
if [[ $PI_MODEL != *"Raspberry Pi 4"* && $PI_MODEL != *"Raspberry Pi 5"* ]]; then
    warn "$PI_MODEL is not a validated target; a Raspberry Pi 4 or 5 is recommended"
fi

[[ $INSTALL_DIR == /* && $INSTALL_DIR != "/" && $INSTALL_DIR != "/opt" ]] || \
    die "--install-dir must be a specific absolute path"
[[ $STATE_DIR == /* && $STATE_DIR != "/" && $STATE_DIR != "/var" ]] || \
    die "--state-dir must be a specific absolute path"
[[ $INSTALL_DIR != *[[:space:]]* && $STATE_DIR != *[[:space:]]* ]] || \
    die "installation paths must not contain whitespace"
[[ $SERVICE_USER =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user: $SERVICE_USER"
[[ $UV_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid uv version: $UV_VERSION"

if [[ -z $DEVICE_ID ]]; then
    DEVICE_ID="$(hostname -s)"
fi
[[ $DEVICE_ID =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "device ID may contain only letters, digits, dots, underscores and hyphens"

for required_file in pyproject.toml uv.lock config/edge.yaml code/dnn_models/yolov8n.onnx; do
    [[ -f "$SOURCE_DIR/$required_file" ]] || \
        die "run this script from a complete BearVision checkout (missing $required_file)"
done

if [[ -d $INSTALL_DIR && ! -f $INSTALL_DIR/pyproject.toml ]]; then
    if find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        die "$INSTALL_DIR is not empty and does not look like a BearVision installation"
    fi
fi

log "Detected $PI_MODEL"
log "Installing Raspberry Pi OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends \
    bluetooth \
    bluez \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    rsync

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    log "Creating unprivileged service account $SERVICE_USER"
    useradd \
        --system \
        --home-dir "$STATE_DIR" \
        --create-home \
        --shell /usr/sbin/nologin \
        "$SERVICE_USER"
fi
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 \
    "$INSTALL_DIR" \
    "$STATE_DIR" \
    "$STATE_DIR/captures" \
    "$STATE_DIR/scratch" \
    "$STATE_DIR/cache"
install -d -o root -g "$SERVICE_GROUP" -m 0750 "$CONFIG_DIR"

log "Copying the active BearVision runtime to $INSTALL_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 \
    "$INSTALL_DIR/config" \
    "$INSTALL_DIR/code/dnn_models"
rsync --archive --delete "$SOURCE_DIR/src/" "$INSTALL_DIR/src/"
install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0644 \
    "$SOURCE_DIR/pyproject.toml" \
    "$SOURCE_DIR/uv.lock" \
    "$SOURCE_DIR/README.md" \
    "$INSTALL_DIR/"
install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0644 \
    "$SOURCE_DIR/config/edge.yaml" \
    "$INSTALL_DIR/config/edge.yaml"
install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0644 \
    "$SOURCE_DIR/code/dnn_models/yolov8n.onnx" \
    "$INSTALL_DIR/code/dnn_models/yolov8n.onnx"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$INSTALL_DIR"

CONFIG_FILE="$CONFIG_DIR/edge.yaml"
CONFIG_WAS_CREATED=false
if [[ ! -e $CONFIG_FILE ]]; then
    install -o root -g "$SERVICE_GROUP" -m 0640 \
        "$SOURCE_DIR/config/edge.yaml" "$CONFIG_FILE"
    CONFIG_WAS_CREATED=true
fi
if [[ $CONFIG_WAS_CREATED == true || $DEVICE_ID_WAS_SET == true ]]; then
    sed -i -E "s|^([[:space:]]*device_id:)[[:space:]].*$|\\1 $DEVICE_ID|" "$CONFIG_FILE"
fi

ENV_FILE="$CONFIG_DIR/bearvision.env"
if [[ ! -e $ENV_FILE ]]; then
    install -o root -g "$SERVICE_GROUP" -m 0640 /dev/null "$ENV_FILE"
    cat > "$ENV_FILE" <<'EOF'
# BearVision service environment. Keep this file out of source control.
# Enable Box uploads in edge.yaml, then set the base64-encoded Box JWT JSON:
# STORAGE_CREDENTIALS_B64=
# STORAGE_CREDENTIALS_B64_2=

# Raspberry Pi uses the FFmpeg build maintained by Raspberry Pi OS.
BEARVISION_FFMPEG=/usr/bin/ffmpeg
BEARVISION_FFPROBE=/usr/bin/ffprobe
EOF
fi

if ! command -v uv >/dev/null 2>&1 || [[ $(uv --version) != "uv $UV_VERSION" ]]; then
    log "Installing pinned uv $UV_VERSION from Astral"
    UV_INSTALLER="$(mktemp)"
    curl --fail --location --proto '=https' --tlsv1.2 \
        "https://astral.sh/uv/$UV_VERSION/install.sh" \
        --output "$UV_INSTALLER"
    env UV_UNMANAGED_INSTALL=/usr/local/bin sh "$UV_INSTALLER"
    rm -f -- "$UV_INSTALLER"
fi

log "Installing Python 3.12 and locked BearVision dependencies"
runuser -u "$SERVICE_USER" -- env \
    HOME="$STATE_DIR" \
    UV_CACHE_DIR="$STATE_DIR/cache/uv" \
    uv sync \
        --directory "$INSTALL_DIR" \
        --locked \
        --python 3.12 \
        --no-dev

UNIT_FILE="/etc/systemd/system/$SERVICE_NAME.service"
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=BearVision edge runtime
Wants=network-online.target bluetooth.service
After=network-online.target bluetooth.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$INSTALL_DIR
Environment=HOME=$STATE_DIR
Environment=PYTHONUNBUFFERED=1
Environment=XDG_CACHE_HOME=$STATE_DIR/cache
EnvironmentFile=-$ENV_FILE
ExecStart=$INSTALL_DIR/.venv/bin/bearvision-edge --config $CONFIG_FILE --capture-dir $STATE_DIR/captures --scratch-dir $STATE_DIR/scratch
Restart=on-failure
RestartSec=5s
TimeoutStopSec=30s
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$STATE_DIR

[Install]
WantedBy=multi-user.target
EOF
chmod 0644 "$UNIT_FILE"

log "Verifying the installed runtime and configuration"
runuser -u "$SERVICE_USER" -- env \
    HOME="$STATE_DIR" \
    "$INSTALL_DIR/.venv/bin/python" -c \
    "from pathlib import Path; from bearvision.config import load_edge_config; load_edge_config(Path('$CONFIG_FILE'))"
/usr/bin/ffmpeg -version >/dev/null
/usr/bin/ffprobe -version >/dev/null

systemctl daemon-reload
systemctl enable bluetooth.service >/dev/null
systemctl enable "$SERVICE_NAME.service" >/dev/null

if [[ $START_SERVICE == true ]]; then
    log "Starting $SERVICE_NAME"
    systemctl restart "$SERVICE_NAME.service"
    if ! systemctl is-active --quiet "$SERVICE_NAME.service"; then
        warn "the service exited; connect the GoPro/BLE hardware and inspect: journalctl -u $SERVICE_NAME -n 100"
    fi
fi

log "Setup complete"
printf '\nConfiguration: %s\nCredentials:   %s\nService:       %s\n' \
    "$CONFIG_FILE" "$ENV_FILE" "$SERVICE_NAME"
if [[ $START_SERVICE == false ]]; then
    printf 'Next step: connect the GoPro and BLE adapter, then run:\n  sudo systemctl start %s\n' \
        "$SERVICE_NAME"
fi
printf 'Logs:\n  journalctl -u %s -f\n' "$SERVICE_NAME"
