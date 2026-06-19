<#
.SYNOPSIS
    Trains the per-device Isolation Forest models against the configured
    MLflow tracking server and Postgres backend.

.DESCRIPTION
    Activates the project virtual environment, ensures the MLflow / Postgres
    environment variables expected by `ml/train_isolation_forest.py` are set,
    and runs the training script from the `ml/` directory so its relative
    imports and paths resolve correctly.

    Run from anywhere; the script resolves the project root from its own
    location:
        .\train_models.ps1

    Override defaults via env vars before invoking, e.g.:
        $env:POSTGRES_DB = 'analytics'
        $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000/'
        .\train_models.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Resolve project root from this script's location so it works from anywhere.
$ProjectRoot  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvActivate = Join-Path $ProjectRoot '.venv\Scripts\Activate.ps1'
$TrainingDir  = Join-Path $ProjectRoot 'ml'
$TrainScript  = Join-Path $TrainingDir 'train_isolation_forest.py'

if (-not (Test-Path $VenvActivate)) {
    throw "Virtual environment activation script not found at: $VenvActivate"
}
if (-not (Test-Path $TrainScript)) {
    throw "Training script not found at: $TrainScript"
}


# Point the training script at the MLflow tracking/registry server. The
# training script reads `MLFLOW_TRACKING_URI` indirectly through its own
# hard-coded default, but we export it here so other tools in the same
# session pick it up too.
if (-not $env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000/'
}


Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
. $VenvActivate

Push-Location $TrainingDir
try {
    Write-Host '=== Training Isolation Forest models ===' -ForegroundColor Cyan
    python .\train_isolation_forest.py
}
finally {
    Pop-Location
}

Write-Host 'Training complete.' -ForegroundColor Green
