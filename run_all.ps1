<#
.SYNOPSIS
    Launches every long-running analytics process in its own PowerShell window.

.DESCRIPTION
    Each component (MLflow UI, Modbus generator, Modbus reader, telemetry
    consumer) is a long-lived process, so they cannot share one terminal.
    This script spawns a new PowerShell window per component, activates the
    project virtual environment in each, and runs the appropriate command
    from the correct working directory.

.NOTES
    Run from the project root:  .\run_all.ps1
    Close each spawned window to stop that component.
#>


$ErrorActionPreference = 'Stop'

# Resolve project root from this script's location so it works from anywhere.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvActivate = Join-Path $ProjectRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path $VenvActivate)) {
    throw "Virtual environment activation script not found at: $VenvActivate"
}


# run the powershell command to run the train_isolation_forest.py script in the ml directory, with the working directory set to ml




function Start-AnalyticsWindow {
    param(
        [Parameter(Mandatory)] [string] $Title,
        [Parameter(Mandatory)] [string] $WorkingDirectory,
        [Parameter(Mandatory)] [string] $Command
    )

    # Build the inner command that runs inside the new PowerShell window:
    #   1. Allow venv activation for this process only
    #   2. Activate the venv
    #   3. cd to the component's working directory
    #   4. Set the window title for easy identification
    #   5. Run the target command
    $inner = @(
        "Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force"
        ". '$VenvActivate'"
        "Set-Location -LiteralPath '$WorkingDirectory'"
        "`$Host.UI.RawUI.WindowTitle = '$Title'"
        "Write-Host '=== $Title ===' -ForegroundColor Cyan"
        $Command
    ) -join '; '

    Write-Host "Starting: $Title" -ForegroundColor Green
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $inner) `
        -WorkingDirectory $WorkingDirectory | Out-Null
}

# --- Components ---------------------------------------------------------------


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

    # Start-AnalyticsWindow `
    #     -Title 'MLflow Server (Postgres)' `
    #     -WorkingDirectory $ProjectRoot `
    #     -Command $mlflowCmd

    # # Give MLflow a moment to bind its port before downstream producers start logging.
    # Start-Sleep -Seconds 10


Start-AnalyticsWindow `
    -Title 'Modbus Generator' `
    -WorkingDirectory (Join-Path $ProjectRoot 'gateway') `
    -Command 'python .\modbus_generator.py'

# Reader connects to the generator, give the generator a moment to come up.
Start-Sleep -Seconds 2

Start-AnalyticsWindow `
    -Title 'Modbus Reader' `
    -WorkingDirectory (Join-Path $ProjectRoot 'gateway') `
    -Command 'python .\modbus_reader.py'

Start-AnalyticsWindow `
    -Title 'Telemetry Consumer' `
    -WorkingDirectory (Join-Path $ProjectRoot 'consumer') `
    -Command 'python .\telemetry_consumer.py'



# Add any additional components here as needed, following the pattern above.
# start-analyticswindow for api server
Start-AnalyticsWindow `
    -Title 'API Server' `
    -WorkingDirectory (Join-Path $ProjectRoot 'api') `
    -Command 'uvicorn main:app --host 0.0.0.0 --port 8000'

# start streamlit app
Start-AnalyticsWindow `
    -Title 'Streamlit App' `
    -WorkingDirectory (Join-Path $ProjectRoot 'app') `
    -Command 'streamlit run streamlit_app.py'


Write-Host ''
Write-Host 'All components launched. Close each window to stop that component.' -ForegroundColor Yellow
