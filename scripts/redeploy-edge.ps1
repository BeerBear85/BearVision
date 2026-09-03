[CmdletBinding()]
param(
    [Parameter()]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$HostName = 'bearvisionedge1.local',

    [Parameter()]
    [ValidatePattern('^[a-z_][a-z0-9_-]*$')]
    [string]$UserName = 'bear',

    [Parameter()]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$DeviceId = 'bearvisionedge1',

    [Parameter()]
    [switch]$CodeOnly,

    [Parameter()]
    [switch]$ConfigureCodeDeploy
)

$ErrorActionPreference = 'Stop'
if ($CodeOnly -and $ConfigureCodeDeploy) {
    throw '-CodeOnly and -ConfigureCodeDeploy are mutually exclusive'
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$deploymentId = [Guid]::NewGuid().ToString('N')
$archivePath = Join-Path ([IO.Path]::GetTempPath()) "bearvision-$deploymentId.tar.gz"
$remoteArchive = "/tmp/bearvision-$deploymentId.tar.gz"
$remoteDirectory = "/tmp/bearvision-$deploymentId"
$destination = "$UserName@$HostName"

$payload = if ($ConfigureCodeDeploy) {
    @('scripts/configure-code-deployment.sh')
}
elseif ($CodeOnly) {
    @(
        'apps/edge-control',
        'scripts/update-raspberry-pi-code.sh',
        'specs/scenarios',
        'src',
        'pyproject.toml',
        'uv.lock'
    )
}
else {
    @(
        'apps/edge-control',
        'code/dnn_models/yolov8n.onnx',
        'config/edge.yaml',
        'README.md',
        'scripts/configure-code-deployment.sh',
        'scripts/setup-raspberry-pi.sh',
        'scripts/update-raspberry-pi-code.sh',
        'specs/scenarios',
        'src',
        'pyproject.toml',
        'uv.lock'
    )
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(ValueFromRemainingArguments)]
        [string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-Host "Checking SSH access to $destination"
    Invoke-NativeCommand ssh '-o' 'BatchMode=yes' '-o' 'ConnectTimeout=8' $destination 'true'

    Write-Host 'Creating Edge deployment archive'
    Invoke-NativeCommand tar '-czf' $archivePath '-C' $repoRoot `
        '--exclude=apps/edge-control/node_modules' `
        '--exclude=apps/edge-control/dist' `
        @payload

    Write-Host "Uploading deployment to $destination"
    Invoke-NativeCommand scp $archivePath "${destination}:$remoteArchive"

    $deploymentCommand = if ($ConfigureCodeDeploy) {
        "sudo bash scripts/configure-code-deployment.sh --deploy-user '$UserName'"
    }
    elseif ($CodeOnly) {
        'bash scripts/update-raspberry-pi-code.sh'
    }
    else {
        "sudo bash scripts/setup-raspberry-pi.sh --device-id '$DeviceId' --deploy-user '$UserName' --start"
    }

    $remoteCommand = @"
set -eu
readonly archive='$remoteArchive'
readonly deploy_dir='$remoteDirectory'
cleanup() {
    rm -f -- "`$archive"
    rm -rf -- "`$deploy_dir"
}
trap cleanup EXIT
install -d -m 0700 "`$deploy_dir"
tar -xzf "`$archive" -C "`$deploy_dir"
cd "`$deploy_dir"
$deploymentCommand
"@

    $mode = if ($ConfigureCodeDeploy) { 'passwordless code deployment' } elseif ($CodeOnly) { 'application code' } else { 'full Edge stack' }
    $privilegeNote = if ($CodeOnly) { 'without sudo' } else { 'sudo may prompt for the remote password' }
    Write-Host "Deploying $mode ($privilegeNote)"
    Invoke-NativeCommand ssh '-tt' $destination $remoteCommand
    Write-Host "Deployment complete: http://$HostName`:4310"
}
finally {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
}
