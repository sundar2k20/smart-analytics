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




$env:POSTGRES_MLFLOW_HOST = 'localhost'
$env:POSTGRES_MLFLOW_PORT = '5432'
$env:POSTGRES_MLFLOW_DB = 'mlflow'
$env:POSTGRES_MLFLOW_USER = 'postgres'
$env:POSTGRES_MLFLOW_PASSWORD = 'Sanadiv@123'  

    # --- MLflow backend store: PostgreSQL ------------------------------------
    # Read Postgres connection details from environment so credentials never
    # live in source. Defaults match database/db_config.py; the only required
    # variable is POSTGRES_PASSWORD.
    if (-not $env:POSTGRES_MLFLOW_PASSWORD) {
        throw "POSTGRES_MLFLOW_PASSWORD env var must be set before launching MLflow."
    }

    $pgHost     = if ($env:POSTGRES_MLFLOW_HOST)   { $env:POSTGRES_MLFLOW_HOST }   else { 'localhost' }
    $pgPort     = if ($env:POSTGRES_MLFLOW_PORT)   { $env:POSTGRES_MLFLOW_PORT }   else { '5432' }
    $pgUser     = if ($env:POSTGRES_MLFLOW_USER)   { $env:POSTGRES_MLFLOW_USER }   else { 'postgres' }
    $mlflowDb   = if ($env:POSTGRES_MLFLOW_DB)       { $env:POSTGRES_MLFLOW_DB }       else { 'mlflow' }
    $pgPassword = [uri]::EscapeDataString($env:POSTGRES_MLFLOW_PASSWORD)

    $backendStoreUri = "postgresql+psycopg2://$pgUser`:$pgPassword@$pgHost`:$pgPort/$mlflowDb"
    $artifactRoot    = (Join-Path $ProjectRoot 'mlartifacts').Replace('\', '/')

    $mlflowCmd = "mlflow server " +
                 "--backend-store-uri '$backendStoreUri' " +
                 "--default-artifact-root 'file:///$artifactRoot' " +
                 "--host 127.0.0.1 --port 5000"

    Start-AnalyticsWindow `
        -Title 'MLflow Server (Postgres)' `
        -WorkingDirectory $ProjectRoot `
        -Command $mlflowCmd

    # Give MLflow a moment to bind its port before downstream producers start logging.
    Start-Sleep -Seconds 10




# Point the training script at the MLflow tracking/registry server. The
# training script reads `MLFLOW_TRACKING_URI` indirectly through its own
# hard-coded default, but we export it here so other tools in the same
# session pick it up too.
if (-not $env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000/'
}

# Warn (don't fail) if the application Postgres password is missing — the
# training script will surface a clearer error if the connection actually
# fails.
if (-not $env:POSTGRES_PASSWORD) {
    Write-Warning "POSTGRES_PASSWORD is not set; train_isolation_forest.py will fail to connect."
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
