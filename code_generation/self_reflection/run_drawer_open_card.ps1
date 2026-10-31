# Full CARD paper loop for drawer-open-v2 (2 introspection rounds, 3 LLM queries):
# step0 -> query0 -> step1 (train) -> query1 -> step2 (train) -> query2 -> step3 (train)
param(
    [string]$ResumeLogPath = "",
    [int]$StartStep = 0
)
$ErrorActionPreference = "Stop"
$repo = "D:\Thesis\Implementation\card-v1.0"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$env:PYTHONPATH = $repo
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MUJOCO_PY_MUJOCO_PATH = "C:\Users\AmirHossein\.mujoco\mujoco210"
$env:MUJOCO_PY_MJKEY_PATH = "C:\Users\AmirHossein\.mujoco\mjkey.txt"
$env:PATH = "C:\Users\AmirHossein\.mujoco\mujoco210\bin;" + $env:PATH
if (-not $env:XAI_KEYS_PATH) {
    $env:XAI_KEYS_PATH = Join-Path $repo "xai_keys.json"
}
if (-not $env:XAI_PROXY) {
    $env:XAI_PROXY = "http://127.0.0.1:10809"
}

$py = "C:\Users\AmirHossein\miniconda3\envs\card\python.exe"
$common = @(
    "--exp_name", "card",
    "--task_name", "drawer-open-v2",
    "--llm_model_name", "grok-4.3",
    "--temperature", "0.7",
    "--eval_freq", "16000",
    "--train_max_steps", "1000000",
    "--train_num", "2",
    "--eval_num", "2",
    "--train_seed_num", "1",
    "--n_eval_episodes", "10",
    "--use_return_check",
    "--threshold", "0.8"
)

function Invoke-Card([string[]]$arguments, [string]$name) {
    Write-Host ""
    Write-Host "======== $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ========"
    & $py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Starting full CARD drawer-open-v2 run"
Write-Host "cwd=$here"
Write-Host "XAI_KEYS_PATH=$($env:XAI_KEYS_PATH)"
Write-Host "XAI_PROXY=$($env:XAI_PROXY)"

if ($ResumeLogPath -ne "") {
    $runLogPath = $ResumeLogPath
    Write-Host "Resuming LOG_PATH=$runLogPath from step $StartStep"
} else {
    $stamp = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
    $runLogPath = Join-Path $here "logs\metaworld\grok-4.3\t=0.7\zero-shot\SAC_1\drawer-open-v2\card\exclude=None\$stamp"
    New-Item -ItemType Directory -Force -Path $runLogPath | Out-Null
    $StartStep = 0
    Invoke-Card (@("metaworld_exp_one_step.py") + $common + @("--step", "0", "--log_path", $runLogPath)) "step0"
}

if (-not (Test-Path -LiteralPath $runLogPath)) {
    throw "LOG_PATH does not exist: $runLogPath"
}
Write-Host "LOG_PATH=$runLogPath"
Set-Content -LiteralPath (Join-Path $here "last_drawer_open_logpath.txt") -Value $runLogPath

if ($StartStep -le 0) {
    $q0 = Join-Path $runLogPath "rl_query_code_0.pkl"
    if (-not (Test-Path -LiteralPath $q0)) { throw "Missing $q0" }
    Invoke-Card @("query_llm_metaworld.py", "--rl_question_file_path", $q0) "query0"
}
if ($StartStep -le 1) {
    Invoke-Card (@("metaworld_exp_one_step.py") + $common + @("--step", "1", "--log_path", $runLogPath)) "step1"
}
if ($StartStep -le 2) {
    $q1 = Join-Path $runLogPath "rl_query_code_1.pkl"
    Invoke-Card @("query_llm_metaworld.py", "--rl_question_file_path", $q1) "query1"
    Invoke-Card (@("metaworld_exp_one_step.py") + $common + @("--step", "2", "--log_path", $runLogPath)) "step2"
}
if ($StartStep -le 3) {
    $q2 = Join-Path $runLogPath "rl_query_code_2.pkl"
    Invoke-Card @("query_llm_metaworld.py", "--rl_question_file_path", $q2) "query2"
    Invoke-Card (@("metaworld_exp_one_step.py") + $common + @("--step", "3", "--log_path", $runLogPath)) "step3"
}

Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "log: $runLogPath"
