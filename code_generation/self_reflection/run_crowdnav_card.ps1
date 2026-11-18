# CARD loop for CrowdNav++ / EvoNav (GST + PPO/SRNN), defaults ON:
# - Dataset gate (results/datasets/crowdnav_ds_v1) + LLM multi-candidate selection
# - CARD+ archive / TPE / SH finalists
# - Short screen then top-K finalist continue
# - Single-process DummyVecEnv (PPO+GST stack; TrainNum>1 does not parallelize yet)
#
# H=20 GST/PPO budget (if short still weak vs H=10): 5x250k screen + 2x1.2M finalists.
# Approx RL ~3.5M env steps. Override with -TrainMaxSteps / -FinalistSteps etc.
param(
    [string]$ResumeLogPath = "",
    [int]$StartStep = 0,
    [int]$TrainRounds = 5,
    [int]$TrainMaxSteps = 250000,
    [int]$FinalistSteps = 1200000,
    [int]$FinalistTopK = 2,
    [switch]$NoFinalists,
    # Mid-train eval ON (needed for SH / success curves). End-of-short also runs FinalEvalEpisodes.
    [int]$EvalFreq = 20000,
    [int]$TrainNum = 1,
    [int]$EvalNum = 1,
    [int]$NEvalEpisodes = 5,
    [int]$FinalEvalEpisodes = 100,
    [int]$LlmCandidates = 5,
    [ValidateSet("auto","dummy","subproc")]
    [string]$VecEnv = "dummy",
    [switch]$Smoke,
    # CARD+ / SH finalists ON by default; pass -NoCardPlus to disable.
    [switch]$NoCardPlus,
    [ValidateSet("warn","gate")]
    [string]$TpeV2Mode = "warn",
    # Offline dataset gate ON by default; pass -NoDatasetRescore to disable.
    [switch]$NoDatasetRescore,
    [ValidateSet("warn","gate")]
    [string]$DatasetMode = "gate",
    [string]$DatasetPath = "",
    [switch]$EvalBest,
    # If dataset missing: fail (default) instead of silently disabling the gate.
    [switch]$AllowMissingDataset,
    # After each CARD step / finalists: commit+push results/ for cross-machine resume.
    [bool]$PushProgress = $true
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
# repo root = card-v1.0 (two levels above self_reflection/)
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
Set-Location $here

if ($TrainRounds -lt 1) { throw "TrainRounds must be >= 1" }

$CardPlus = -not $NoCardPlus

$env:PYTHONPATH = $repo
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:XAI_KEYS_PATH) {
    $env:XAI_KEYS_PATH = Join-Path $repo "xai_keys.json"
}
# Do not force a local proxy; set XAI_PROXY yourself if you need one.

if ($Smoke) {
    $TrainMaxSteps = 2000
    $FinalistSteps = 4000
    $EvalFreq = 1000
    $NEvalEpisodes = 2
    $FinalEvalEpisodes = 4
}

if ($env:CARD_PYTHON -and (Test-Path -LiteralPath $env:CARD_PYTHON)) {
    $py = $env:CARD_PYTHON
} else {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "python not found. Set CARD_PYTHON to your card env interpreter." }
    $py = $cmd.Source
}

# Fastest measured on this machine: CPU + 4 torch threads (ORCA-bound; GPU optional via CARD_DEVICE=cuda)
if (-not $env:CARD_DEVICE) { $env:CARD_DEVICE = "cpu" }

Write-Host "REPO=$repo"
Write-Host "PYTHON=$py"
$gstFlag = if ($env:CARD_GST -and $env:CARD_GST -match '^(1|true|yes|on)$') { "GST=on(inferred)" } else { "GST=off(const_vel)" }
Write-Host "Stack: CrowdNav++ PPO/selfAttn SRNN + $gstFlag | env H=20 robot_visible=False"
Write-Host "Device: CARD_DEVICE=$($env:CARD_DEVICE) torch_threads=4"
Write-Host "Features: CardPlus/SH=$CardPlus DatasetGate=$(-not $NoDatasetRescore) Finalists=$(-not $NoFinalists)"

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
    "--torch_threads", "4",
    "--sub_reward_eval_mult", "4",
    "--train_seed_num", "1",
    "--n_eval_episodes", "$NEvalEpisodes",
    "--final_eval_episodes", "$FinalEvalEpisodes",
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
    Write-Host "CARD+ enabled: TPE v2 ($TpeV2Mode) + sub-reward stats + archive/rollback + SH finalists"
}

# Optimal dataset gate: keep only rewards that separate success/collision/timeout.
if (-not $NoDatasetRescore) {
    if ($DatasetPath -eq "") {
        $DatasetPath = Join-Path $repo "results\datasets\crowdnav_ds_v1"
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
            "--dataset_min_composite", "0.75",
            "--dataset_min_anti_hack", "0.65",
            "--dataset_min_separation", "0.70",
            "--llm_candidates", "$LlmCandidates",
            "--llm_candidate_retries", "4"
        )
        Write-Host "Dataset gate ON ($DatasetMode): $DatasetPath (LLM $LlmCandidates candidates -> best composite out)"
    } else {
        $msg = "Dataset not found at $DatasetPath. Collect first: python -m card_plus.dataset.collect --out results/datasets/crowdnav_ds_v1 ..."
        if ($AllowMissingDataset) {
            Write-Host "WARNING: $msg — gate disabled (-AllowMissingDataset)"
        } else {
            throw $msg
        }
    }
}
Write-Host "TrainNum=$TrainNum EvalNum=$EvalNum VecEnv=$VecEnv CardPlus=$CardPlus"
$continueSteps = [Math]::Max(0, $FinalistSteps - $TrainMaxSteps)
$approxShort = $TrainRounds * $TrainMaxSteps
$approxFinal = $FinalistTopK * $continueSteps
Write-Host "LEAN budget: short=${TrainRounds}x${TrainMaxSteps}=$approxShort; finalists=${FinalistTopK}x+${continueSteps}=$approxFinal; total_RL≈$($approxShort+$approxFinal)"
Write-Host "eval: mid_train every ${EvalFreq} steps x ${NEvalEpisodes}eps (SH); end_of_short=${FinalEvalEpisodes}eps; llm_cand=$LlmCandidates; NoFinalists=$NoFinalists"

function Invoke-Card([string[]]$arguments, [string]$name) {
    # sync_crowdnav_progress.ps1 Set-Locations to repo root; always restore before python.
    Set-Location $here
    $argv = [System.Collections.Generic.List[string]]::new()
    foreach ($a in $arguments) { [void]$argv.Add([string]$a) }
    if ($argv.Count -gt 0 -and $argv[0] -like "*.py" -and -not [System.IO.Path]::IsPathRooted($argv[0])) {
        $argv[0] = Join-Path $here $argv[0]
    }
    Write-Host ""
    Write-Host "======== $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ========"
    & $py @($argv.ToArray())
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

function Sync-Progress([string]$message) {
    if (-not $PushProgress) { return }
    $syncPs1 = Join-Path $here "sync_crowdnav_progress.ps1"
    Write-Host "---- git sync: $message ----"
    try {
        & $syncPs1 -Message $message -LogPath $runLogPath
    } catch {
        Write-Host "WARNING: progress sync failed (training continues): $_"
    } finally {
        Set-Location $here
    }
}

Write-Host "Starting CARD CrowdNav++ circle-crossing-v0 run"
Write-Host "cwd=$here TrainRounds=$TrainRounds TrainMaxSteps=$TrainMaxSteps Smoke=$Smoke PushProgress=$PushProgress"
Write-Host "XAI_KEYS_PATH=$($env:XAI_KEYS_PATH)"
Write-Host "XAI_PROXY=$($env:XAI_PROXY)"

$runsRoot = Join-Path $repo "results\runs\crowdnav"
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

if ($ResumeLogPath -ne "") {
    $runLogPath = $ResumeLogPath
    Write-Host "Resuming LOG_PATH=$runLogPath from step $StartStep"
} else {
    $stamp = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
    # Portable path: pushed under results/ so another machine can git pull + resume.
    $runLogPath = Join-Path $runsRoot "PPO_GST_H20_$stamp"
    New-Item -ItemType Directory -Force -Path $runLogPath | Out-Null
    $StartStep = 0
    Invoke-Card (@("crowdnav_exp_one_step.py") + $common + @("--step", "0", "--log_path", $runLogPath)) "step0"
    Sync-Progress "crowdnav card step0 init $stamp"
}

if (-not (Test-Path -LiteralPath $runLogPath)) {
    throw "LOG_PATH does not exist: $runLogPath"
}
Write-Host "LOG_PATH=$runLogPath"
Set-Content -LiteralPath (Join-Path $here "last_crowdnav_logpath.txt") -Value $runLogPath
Set-Content -LiteralPath (Join-Path $runsRoot "LAST_LOGPATH.txt") -Value $runLogPath

# For train round k=1..TrainRounds:
#   query(k-1) generates code_(k-1), then step k trains it (short budget).
# Resume: StartStep=1 skips query0 and reruns step1 (matches drawer-open).
for ($k = 1; $k -le $TrainRounds; $k++) {
    $queryStep = $k - 1
    if ($StartStep -le $queryStep) {
        $qPath = Join-Path $runLogPath "rl_query_code_$queryStep.pkl"
        if (-not (Test-Path -LiteralPath $qPath)) { throw "Missing $qPath" }
        $respPath = Join-Path $runLogPath "llm_response_code_$queryStep.pkl"
        $codePath = Join-Path $runLogPath "code_$queryStep\specific_$queryStep.py"
        if (Test-Path -LiteralPath $respPath) {
            Write-Host "Found $respPath -> skip query$queryStep"
        } elseif (Test-Path -LiteralPath $codePath) {
            # Winner code already saved (e.g. crash after save_code); skip costly LLM redo.
            Write-Host "Found $codePath -> skip query$queryStep (reward already generated)"
        } else {
            Invoke-Card @("query_llm_crowdnav.py", "--rl_question_file_path", $qPath) "query$queryStep"
            Sync-Progress "crowdnav card query$queryStep"
        }
    }
    if ($StartStep -le $k) {
        $stepArgs = @("crowdnav_exp_one_step.py") + $common + @("--step", "$k", "--log_path", $runLogPath)
        # If train already finished and checkpoint exists, skip re-training.
        $ckpt = Join-Path $runLogPath "checkpoint_$k.pkl"
        $modelPt = Join-Path $runLogPath "code_$($k - 1)\seed_1\tensorboard_log\latest_model.pt"
        $evalNpz = Join-Path $runLogPath "code_$($k - 1)\seed_1\tensorboard_log\eval\evaluations.npz"
        $finalEval = Join-Path $runLogPath "code_$($k - 1)\seed_1\tensorboard_log\eval\final_eval.json"
        $skippedTrain = $false
        if (Test-Path -LiteralPath $ckpt) {
            $stepArgs += "--checkpoint"
            $skippedTrain = $true
            Write-Host "Found $ckpt -> loading with --checkpoint (skip train)"
        } elseif ((Test-Path -LiteralPath $modelPt) -and ((Test-Path -LiteralPath $evalNpz) -or (Test-Path -LiteralPath $finalEval))) {
            # Train finished (eval artifacts present): skip retrain, finish collect/feedback.
            $stepArgs += "--load_model"
            Write-Host "Found completed $modelPt + eval -> --load_model (skip train)"
        } elseif (Test-Path -LiteralPath $modelPt) {
            # Mid-train interrupt: do NOT --load_model; Python resumes/heals corrupt .pt.
            Write-Host "Found incomplete $modelPt (no eval yet) -> resume/retrain in Python"
        }
        Invoke-Card $stepArgs "step$k"
        if (-not $skippedTrain) {
            Sync-Progress "crowdnav card step$k short-train"
        }
    }
}

# Across-step SH: top-K short-trained versions get a long final train.
if (-not $NoFinalists) {
    if (-not $CardPlus) {
        Write-Host "WARNING: finalists need CARD+ archive; skipping finalist phase (remove -NoCardPlus)"
    } else {
        $finReport = Join-Path $runLogPath "finalists\report.json"
        if (Test-Path -LiteralPath $finReport) {
            Write-Host "Found $finReport -> skip finalists"
        } else {
            $finArgs = @(
                "crowdnav_finalists.py",
                "--log_path", "$runLogPath",
                "--top_k", "$FinalistTopK",
                "--train_max_steps", "$FinalistSteps",
                "--short_train_steps", "$TrainMaxSteps",
                "--final_eval_seed", "1001",
                "--final_eval_episodes", "50",
                "--eval_freq", "$EvalFreq",
                "--train_num", "$TrainNum",
                "--eval_num", "$EvalNum",
                "--vec_env", "$VecEnv",
                "--torch_threads", "4",
                "--sub_reward_eval_mult", "2",
                "--max_episode_steps", "200",
                "--n_eval_episodes", "$NEvalEpisodes",
                "--trajectory_num", "20",
                "--task_name", "circle-crossing-v0",
                "--benchmark_name", "crowdnav",
                "--llm_model_name", "grok-4.3",
                "--exp_name", "card",
                "--seed", "1"
            )
            Invoke-Card $finArgs "finalists_${FinalistTopK}x${FinalistSteps}"
            Sync-Progress "crowdnav card finalists done"
        }
    }
}

if ($EvalBest) {
    if (-not $CardPlus) { Write-Host "WARNING: -EvalBest without CARD+ (need card_plus_best/specific.py)" }
    Invoke-Card @(
        "-m", "card_plus.eval_best",
        "--log_path", $runLogPath,
        "--benchmark", "crowdnav",
        "--eval_seed", "1001",
        "--selection_seed", "1"
    ) "eval_best"
}

Sync-Progress "crowdnav card DONE"
Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "log: $runLogPath"
Write-Host "Resume on another machine:"
Write-Host "  git pull"
Write-Host "  copy xai_key (not in git)"
Write-Host "  .\run_crowdnav_card.ps1 -ResumeLogPath <log> -StartStep N"
$resultsHint = Join-Path $repo "results\crowdnav"
if (Test-Path -LiteralPath $resultsHint) {
    $latestResult = Get-ChildItem -LiteralPath $resultsHint -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestResult) {
        Write-Host "curated results: $($latestResult.FullName)"
    }
}
