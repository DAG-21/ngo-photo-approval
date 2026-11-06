# Runs every test suite of the project with the project's own Python environment. No model is called and nothing is sent anywhere.
$root = $PSScriptRoot
$py = Join-Path $root 'image-classifier\.venv\Scripts\python.exe'
foreach ($folder in 'layered-engine', 'agent-platform', 'holistic-judge-trial', 'image-classifier') {
    $dir = Join-Path $root $folder
    if (-not (Test-Path (Join-Path $dir 'tests'))) { continue }
    Write-Host "== $folder"
    Push-Location $dir
    & $py -m pytest tests -q -p no:cacheprovider
    Pop-Location
}
