# CARD loop for CrowdNav++ circle-crossing-v0:
# step0 -> query0 -> step1 (train) -> query1 -> ... -> query(N-1) -> stepN (train)
# TrainRounds = number of SAC train stages (default 5 => ends at step5).
param(
    [string]$ResumeLogPath = "",
    [int]$StartStep = 0,
    [int]$TrainRounds = 5,
    [int]$TrainMaxSteps = 200000,
    [int]$EvalFreq = 5000,
    [int]$TrainNum = 4,
    [int]$EvalNum = 4,
    [ValidateSet("auto","dummy","subproc")]
    [string]$VecEnv = "auto",
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$repo = "D:\Thesis\Implementation\card-v1.0"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if ($TrainRounds -lt 1) { throw "TrainRounds must be >= 1" }

$env:PYTHONPATH = $repo
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:XAI_KEYS_PATH) {
    $env:XAI_KEYS_PATH = Join-Path $repo "xai_keys.json"
}
if (-not $env:XAI_PROXY) {
    $env:XAI_PROXY = "http://127.0.0.1:10809"
}

if ($Smoke) {
    $TrainMaxSteps = 2000
    $EvalFreq = 1000
}

$py = "C:\Users\AmirHossein\miniconda3\envs\card\python.exe"
$common = @(
    "--exp_name", "card",
    "--task_name", "circle-crossing-v0",
    "--llm_model_name", "grok-4.3",
    "--temperature", "0.7",
    "--eval_freq", "$EvalFreq",
    "--train_max_steps", "$TrainMaxSteps",
    "--max_episode_steps", "200",
    "--train_num", "$TrainNum",
    "--eval_num", "$EvalNum",
    "--vec_env", "$VecEnv",
    "--torch_threads", "1",
    "--sub_reward_eval_mult", "2",
    "--train_seed_num", "1",
    "--n_eval_episodes", "10",
    "--trajectory_num", "20",
    "--log_interval", "10",
    "--use_return_check",
    "--threshold", "0.8"
)
Write-Host "TrainNum=$TrainNum EvalNum=$EvalNum VecEnv=$VecEnv"

function Invoke-Card([string[]]$arguments, [string]$name) {
    Write-Host ""
    Write-Host "======== $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ========"
    & $py @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Starting CARD CrowdNav++ circle-crossing-v0 run"
Write-Host "cwd=$here TrainRounds=$TrainRounds TrainMaxSteps=$TrainMaxSteps Smoke=$Smoke"
Write-Host "XAI_KEYS_PATH=$($env:XAI_KEYS_PATH)"
Write-Host "XAI_PROXY=$($env:XAI_PROXY)"

if ($ResumeLogPath -ne "") {
    $runLogPath = $ResumeLogPath
    Write-Host "Resuming LOG_PATH=$runLogPath from step $StartStep"
} else {
    $stamp = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
    $runLogPath = Join-Path $here "logs\crowdnav\grok-4.3\t=0.7\zero-shot\SAC_1\circle-crossing-v0\card\exclude=None\$stamp"
    New-Item -ItemType Directory -Force -Path $runLogPath | Out-Null
    $StartStep = 0
    Invoke-Card (@("crowdnav_exp_one_step.py") + $common + @("--step", "0", "--log_path", $runLogPath)) "step0"
}

if (-not (Test-Path -LiteralPath $runLogPath)) {
    throw "LOG_PATH does not exist: $runLogPath"
}
Write-Host "LOG_PATH=$runLogPath"
Set-Content -LiteralPath (Join-Path $here "last_crowdnav_logpath.txt") -Value $runLogPath

for ($k = 1; $k -le $TrainRounds; $k++) {
    $queryStep = $k - 1
    if ($StartStep -le $queryStep) {
        $qPath = Join-Path $runLogPath "rl_query_code_$queryStep.pkl"
        if (-not (Test-Path -LiteralPath $qPath)) { throw "Missing $qPath" }
        $respPath = Join-Path $runLogPath "llm_response_code_$queryStep.pkl"
        if (Test-Path -LiteralPath $respPath) {
            Write-Host "Found $respPath -> skip query$queryStep"
        } else {
            Invoke-Card @("query_llm_crowdnav.py", "--rl_question_file_path", $qPath) "query$queryStep"
        }
    }
    if ($StartStep -le $k) {
        $stepArgs = @("crowdnav_exp_one_step.py") + $common + @("--step", "$k", "--log_path", $runLogPath)
        $ckpt = Join-Path $runLogPath "checkpoint_$k.pkl"
        if (Test-Path -LiteralPath $ckpt) {
            $stepArgs += "--checkpoint"
            Write-Host "Found $ckpt -> loading with --checkpoint (skip train)"
        }
        Invoke-Card $stepArgs "step$k"
    }
}

Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "log: $runLogPath"
