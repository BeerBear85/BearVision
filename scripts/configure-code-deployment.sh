#!/usr/bin/env bash

# Configure least-privilege, passwordless application-code deployment.
# This one-time bootstrap must run as root; routine code updates must not.

set -Eeuo pipefail

readonly DEFAULT_INSTALL_DIR="/opt/bearvision"
readonly DEFAULT_SERVICE_USER="bearvision"
readonly DEFAULT_CONTROL_SERVICE_NAME="bearvision-edge-control"
readonly SUDOERS_FILE="/etc/sudoers.d/bearvision-code-deploy"

INSTALL_DIR="$DEFAULT_INSTALL_DIR"
SERVICE_USER="$DEFAULT_SERVICE_USER"
CONTROL_SERVICE_NAME="$DEFAULT_CONTROL_SERVICE_NAME"
DEPLOY_USER=""
SUDOERS_CANDIDATE=""

usage() {
    cat <<'EOF'
Usage: sudo bash scripts/configure-code-deployment.sh --deploy-user USER [options]

Options:
  --deploy-user USER   Login allowed to deploy BearVision application code.
  --install-dir PATH   Application directory (default: /opt/bearvision).
  --service-user USER  Runtime account (default: bearvision).
  --service-name NAME  Edge Control systemd service (default: bearvision-edge-control).
  -h, --help           Show this help.
EOF
}

log() {
    printf '[BearVision code deployment] %s\n' "$*"
}

die() {
    printf '[BearVision code deployment] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n $SUDOERS_CANDIDATE && -e $SUDOERS_CANDIDATE ]]; then
        rm -f -- "$SUDOERS_CANDIDATE"
    fi
}

trap cleanup EXIT

require_value() {
    local option=$1
    local value=${2-}
    [[ -n $value ]] || die "$option requires a value"
}

while (($# > 0)); do
    case "$1" in
        --deploy-user)
            require_value "$1" "${2-}"
            DEPLOY_USER=$2
            shift 2
            ;;
        --install-dir)
            require_value "$1" "${2-}"
            INSTALL_DIR=$2
            shift 2
            ;;
        --service-user)
            require_value "$1" "${2-}"
            SERVICE_USER=$2
            shift 2
            ;;
        --service-name)
            require_value "$1" "${2-}"
            CONTROL_SERVICE_NAME=$2
            shift 2
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

[[ $EUID -eq 0 ]] || die "run this one-time bootstrap as root (for example, with sudo)"
[[ $DEPLOY_USER =~ ^[a-z_][a-z0-9_-]*$ && $DEPLOY_USER != root ]] || \
    die "--deploy-user must name a non-root login"
[[ $SERVICE_USER =~ ^[a-z_][a-z0-9_-]*$ ]] || die "invalid service user: $SERVICE_USER"
[[ $CONTROL_SERVICE_NAME =~ ^[A-Za-z0-9_.@-]+$ ]] || \
    die "invalid service name: $CONTROL_SERVICE_NAME"
[[ $INSTALL_DIR == /* && $INSTALL_DIR != / && $INSTALL_DIR != /opt ]] || \
    die "--install-dir must be a specific absolute path"
[[ $INSTALL_DIR != *[[:space:]]* ]] || die "installation path must not contain whitespace"

id "$DEPLOY_USER" >/dev/null 2>&1 || die "deploy user does not exist: $DEPLOY_USER"
id "$SERVICE_USER" >/dev/null 2>&1 || die "service user does not exist: $SERVICE_USER"
[[ -x /usr/bin/systemctl ]] || die "/usr/bin/systemctl is required"
command -v visudo >/dev/null 2>&1 || die "visudo is required"

UNIT_USER="$(/usr/bin/systemctl show --property User --value "$CONTROL_SERVICE_NAME.service")"
UNIT_GROUP="$(/usr/bin/systemctl show --property Group --value "$CONTROL_SERVICE_NAME.service")"
[[ $UNIT_USER == "$SERVICE_USER" ]] || \
    die "$CONTROL_SERVICE_NAME.service must run as $SERVICE_USER, not $UNIT_USER"
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"
[[ $UNIT_GROUP == "$SERVICE_GROUP" ]] || \
    die "$CONTROL_SERVICE_NAME.service must run as group $SERVICE_GROUP, not $UNIT_GROUP"

CODE_DIRECTORIES=(
    "$INSTALL_DIR/src"
    "$INSTALL_DIR/apps/edge-control"
    "$INSTALL_DIR/specs/scenarios"
)
for directory in "${CODE_DIRECTORIES[@]}"; do
    [[ -d $directory ]] || die "missing application directory: $directory"
done

log "Granting $DEPLOY_USER ownership of deployable application code"
usermod --append --groups "$SERVICE_GROUP" "$DEPLOY_USER"
chown -R "$DEPLOY_USER:$SERVICE_GROUP" "${CODE_DIRECTORIES[@]}"
chmod -R u=rwX,g=rX,o= "${CODE_DIRECTORIES[@]}"
find "${CODE_DIRECTORIES[@]}" -type d -exec chmod g+s {} +

log "Allowing passwordless restart of only $CONTROL_SERVICE_NAME.service"
SUDOERS_CANDIDATE="$(mktemp)"
printf '%s ALL=(root) NOPASSWD: /usr/bin/systemctl restart %s.service\n' \
    "$DEPLOY_USER" "$CONTROL_SERVICE_NAME" > "$SUDOERS_CANDIDATE"
chmod 0440 "$SUDOERS_CANDIDATE"
visudo -cf "$SUDOERS_CANDIDATE" >/dev/null
install -o root -g root -m 0440 "$SUDOERS_CANDIDATE" "$SUDOERS_FILE"

log "Passwordless code deployment configured for $DEPLOY_USER"
