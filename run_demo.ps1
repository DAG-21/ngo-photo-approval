# Starts the review demo at http://127.0.0.1:8100 with the project's own Python environment and the embedding models kept in this folder.
#   .\run_demo.ps1                    offline: no model is called; only the stored pilot answers are used (a photograph outside the pilot is refused)
#   .\run_demo.ps1 -Live              LIVE with Claude Haiku 5.5: every uploaded photograph is sent to Anthropic and reviewers' reasons to Groq (this costs money)
#   .\run_demo.ps1 -Live -Vision gemini   LIVE with Gemini (gemini-2.5-flash-lite) instead: photographs go to Google
# Haiku needs, in image-classifier\.env: ANTHROPIC_API_KEY and NGO_ALLOW_ANTHROPIC_INFERENCE=true, and the anthropic package in the project's environment
#   (image-classifier\.venv\Scripts\python.exe -m pip install anthropic). Without them the start stops with a message saying what is missing; nothing is sent.
# A Claude Pro subscription does not include API calls: the API is billed separately from the Anthropic Console.
# Stop it with Ctrl+C. Logging in again is needed after every start. Logins: agent-platform\demo_data\fresh\credentials.txt
param([switch]$Live, [ValidateSet('haiku', 'gemini')][string]$Vision = 'haiku', [string]$RulesProfile = 'layered_tuned_v1')
$root = $PSScriptRoot
$env:HF_HUB_CACHE = Join-Path $root 'models\huggingface\hub'   # the embedding models for "similar photographs" live in this folder
$env:HF_HUB_OFFLINE = '1'                                       # nothing is ever downloaded
$py = Join-Path $root 'image-classifier\.venv\Scripts\python.exe'
Set-Location (Join-Path $root 'agent-platform')
# the demo organisation still uses layered_tuned_v1; layered_final_v1 (the owner's reviewed rules) is not adopted by the demo yet
$args_ = @('scripts/serve_fresh.py', '--profile', $RulesProfile)
if ($Live) { $args_ += @('--live', '--vision', $Vision, '--llm-gaps', '--gap-llm', 'groq', '--llm-rules', '--llm-summary') }
& $py @args_
