[CmdletBinding()]
param(
  [string]$Project = "visionid",
  [string]$FrontendDirectory = "$PSScriptRoot\..\frontend",
  [string]$ProductionUrl = "https://visionid-murex.vercel.app",
  [int]$HealthTimeoutSeconds = 90,
  [int]$TunnelTimeoutSeconds = 45
)

$ErrorActionPreference = "Stop"
$tokenName = "VERCEL_TOKEN"
$token = [Environment]::GetEnvironmentVariable($tokenName)
if ([string]::IsNullOrWhiteSpace($token)) {
  throw "$tokenName is not set. Set it in the current PowerShell session before running this script."
}

function Stop-OnFailure([string]$Message) { throw $Message }
function Get-Health([string]$Url) {
  try {
    return Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
  } catch { return $null }
}

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$uvicorn = Start-Process -FilePath "uvicorn" -ArgumentList "api.main:app --host 0.0.0.0 --port 8000" -WorkingDirectory $repoRoot -RedirectStandardOutput "$env:TEMP\visionid-uvicorn.out.log" -RedirectStandardError "$env:TEMP\visionid-uvicorn.err.log" -PassThru
try {
  $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
  do {
    Start-Sleep -Seconds 2
    $health = Get-Health "http://localhost:8000/api/health"
  } while ($null -eq $health -and (Get-Date) -lt $deadline)
  if ($null -eq $health -or $health.StatusCode -ne 200) { Stop-OnFailure "Backend health check failed. See $env:TEMP\visionid-uvicorn.err.log" }

  $tunnelOut = "$env:TEMP\visionid-cloudflared.out.log"
  $tunnelErr = "$env:TEMP\visionid-cloudflared.err.log"
  Remove-Item $tunnelOut, $tunnelErr -Force -ErrorAction SilentlyContinue
  $cloudflared = Start-Process -FilePath "cloudflared" -ArgumentList "tunnel --url http://localhost:8000" -RedirectStandardOutput $tunnelOut -RedirectStandardError $tunnelErr -PassThru
  $tunnelLogs = @($tunnelOut, $tunnelErr)
  $deadline = (Get-Date).AddSeconds($TunnelTimeoutSeconds)
  $tunnelUrl = $null
  do {
    Start-Sleep -Seconds 2
    $tunnelUrl = Select-String -Path $tunnelLogs -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue | ForEach-Object { $_.Matches.Value } | Select-Object -First 1
  } while ([string]::IsNullOrWhiteSpace($tunnelUrl) -and (Get-Date) -lt $deadline)
  if ([string]::IsNullOrWhiteSpace($tunnelUrl)) { Stop-OnFailure "Cloudflare tunnel URL was not found. See $tunnelOut and $tunnelErr" }

  $env:VERCEL_TOKEN = $token
  Push-Location $FrontendDirectory
  try {
    $envLines = & vercel env ls $Project --token $token 2>&1
    if ($LASTEXITCODE -ne 0) { Stop-OnFailure "Could not access Vercel project '$Project': $($envLines -join ' ')" }
    & vercel env rm VITE_API_BASE_URL production --yes --token $token --project $Project
    if ($LASTEXITCODE -ne 0) { Write-Warning "VITE_API_BASE_URL was not removed; continuing with add/overwrite." }
    $tunnelUrl | & vercel env add VITE_API_BASE_URL production --token $token --project $Project
    if ($LASTEXITCODE -ne 0) { Stop-OnFailure "Could not update VITE_API_BASE_URL." }
    $deploymentUrl = (& vercel deploy --prod --token $token --project $Project --yes | Select-Object -Last 1).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($deploymentUrl)) { Stop-OnFailure "Vercel deployment failed." }
  } finally { Pop-Location }

  $live = Get-Health "$ProductionUrl/api/health"
  if ($null -eq $live -or $live.StatusCode -ne 200) { Stop-OnFailure "Frontend/backend end-to-end health check failed at $ProductionUrl/api/health" }
  Write-Host "VisionID startup completed." -ForegroundColor Green
  Write-Host "Tunnel: $tunnelUrl"
  Write-Host "Deployment: $deploymentUrl"
  Write-Host "Status: backend and production health checks passed."
} catch {
  Write-Error $_
  exit 1
}

Write-Host "uvicorn PID: $($uvicorn.Id); cloudflared PID: $($cloudflared.Id)"
