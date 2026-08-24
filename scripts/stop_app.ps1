$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $projectRoot ".app-pids.json"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No running instance was recorded for this project."
    exit 0
}

$record = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
$targets = @(
    @{ Id = [int]$record.streamlit; Marker = "streamlit_app.py" },
    @{ Id = [int]$record.worker; Marker = "worker.py" }
)

foreach ($target in $targets) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($target.Id)" -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        continue
    }
    if ($process.CommandLine -notmatch [regex]::Escape($projectRoot) -or
        $process.CommandLine -notmatch [regex]::Escape($target.Marker)) {
        Write-Error "PID $($target.Id) does not belong to this project and was not stopped."
        continue
    }
    & taskkill.exe /PID $target.Id /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to stop PID $($target.Id)."
    }
}

Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "This project's Streamlit server and worker have been stopped."
