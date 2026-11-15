# CARD loop for CrowdNav++ circle-crossing-v0:
# step0 -> query0 -> step1 (train) -> query1 -> ... -> query(N-1) -> stepN (train)
# TrainRounds = number of SAC train stages (default 5 => ends at step5).
param(
    [string]$ResumeLogPath = "",
    [int]$StartStep = 0,
    [int]$TrainRounds = 5,
    [int]$TrainMaxSteps = 100000,
    [int]$EvalFreq = 5000,
    [int]$TrainNum = 4,
    [int]$EvalNum = 4,
    [ValidateSet("auto","dummy","subproc")]
    [string]$VecEnv = "auto",
    [switch]$Smoke,
    [switch]$CardPlus,
    [ValidateSet("warn","gate")]
    [string]$TpeV2Mode = "warn",
    # Offline dataset gate (optimal defaults). On unless -NoDatasetRescore.
    [switch]$NoDatasetRescore,
    [ValidateSet("warn","gate")]
    [string]$DatasetMode = "gate",
    [string]$DatasetPath = "",
    [switch]$EvalBest
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# repo root = .../Card-plus (two levels above self_reflection/)
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
Set-Location $here

if ($TrainRounds -lt 1) { throw "TrainRounds must be >= 1" }

$env:PYTHONPATH = $repo
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:XAI_KEYS_PATH) {
    $env:XAI_KEYS_PATH = Join-Path $repo "xai_keys.json"
}
# Do not force a local proxy; set XAI_PROXY yourself if you need one.

if ($Smoke) {
    $TrainMaxSteps = 2000
    $EvalFreq = 1000
}

if ($env:CARD_PYTHON -and (Test-Path -LiteralPath $env:CARD_PYTHON)) {
    $py = $env:CARD_PYTHON
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "python not found. Set CARD_PYTHON to your card env interpreter." }
    $py = $cmd.Source
}
Write-Host "REPO=$repo"
Write-Host "PYTHON=$py"
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
    "--threshold", "0.8",
    "--card_plus_max_step", "$TrainRounds"
)
if ($CardPlus) {
    $common += @(
        "--use_tpe_v2",
        "--tpe_v2_mode", "$TpeV2Mode",
        "--use_subreward_stats",
        "--use_archive_rollback"
    )
    Write-Host "CARD+ enabled: TPE v2 ($TpeV2Mode) + sub-reward stats + archive/rollback"
}
# Optimal dataset gate: keep only rewards that separate success/collision/timeout.
if (-not $NoDatasetRescore) {
    if ($DatasetPath -eq "") {
        $DatasetPath = Join-Path $repo "logs\crowdnav_ds_v1"
    }
    if (Test-Path -LiteralPath $DatasetPath) {
        $common += @(
            "--use_dataset_rescore",
            "--dataset_path", "$DatasetPath",
            "--dataset_mode", "$DatasetMode",
            "--dataset_min_success_collided", "0.85",
            "--dataset_min_success_timeout", "0.80",
            "--dataset_min_timeout_collided", "0.60",
            "--dataset_min_progress_auc", "0.75",
            "--dataset_min_composite", "0.65",
            "--dataset_min_anti_hack", "0.55",
            "--dataset_min_separation", "0.55",
            "--llm_candidates", "3",
            "--llm_candidate_retries", "5"
        )
        Write-Host "Dataset gate ON ($DatasetMode): $DatasetPath (LLM 3 candidates, pick best)"
    } else {
        Write-Host "WARNING: dataset not found at $DatasetPath — gate disabled"
    }
}
Write-Host "TrainNum=$TrainNum EvalNum=$EvalNum VecEnv=$VecEnv CardPlus=$CardPlus"

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

# For train round k=1..TrainRounds:
#   query(k-1) generates code_(k-1), then step k trains it.
# Resume: StartStep=1 skips query0 and reruns step1 (matches drawer-open).
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
        # If train already finished and checkpoint exists, skip re-training.
        $ckpt = Join-Path $runLogPath "checkpoint_$k.pkl"
        if (Test-Path -LiteralPath $ckpt) {
            $stepArgs += "--checkpoint"
            Write-Host "Found $ckpt -> loading with --checkpoint (skip train)"
        }
        Invoke-Card $stepArgs "step$k"
    }
}

if ($EvalBest) {
    if (-not $CardPlus) { Write-Host "WARNING: -EvalBest without -CardPlus (need card_plus_best/specific.py)" }
    Invoke-Card @(
        "-m", "card_plus.eval_best",
        "--log_path", $runLogPath,
        "--benchmark", "crowdnav",
        "--eval_seed", "1001",
        "--selection_seed", "1"
    ) "eval_best"
}

Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "log: $runLogPath"
