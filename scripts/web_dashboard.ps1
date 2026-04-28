$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..\web")
try {
    npm install --silent
    npm run dev
} finally {
    Pop-Location
}
