# CARD loop for CrowdNav. Two backends:
#   -Algo sac  : flat obs + SB3 SAC (no GST)
#                H=10: 5x150k + 2x550k finalists (legacy ~92% recipe)
#                H>=15: 5x250k + 2x800k; eval 10k/10 mid, held-out 100
#   -Algo ppo  : CrowdNav++ selfAttn SRNN (+ optional CARD_GST=1)
#                default budget 5x250k short + 2x1.2M finalists, H=20 unless CARD_HUMAN_NUM set
#
# Dataset gate + CARD+ / SH finalists ON by default.
param(
    [ValidateSet("sac","ppo")]
    [string]$Algo = "ppo",
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
    # G2 learned judge (Phase-2 No-Go redesign). Empty = off; set path or -UseG2Judge.
    [string]$LearnedJudge = "",
    [switch]$UseG2Judge
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

# SAC flat profile. H=10 matches the ~92% run; H>=15 uses a larger train budget.
# Eval: 10k freq x 10 eps (same mid density as the proven run); held-out 100 eps.
$env:CARD_ALGO = $Algo
if ($Algo -eq "sac") {
    if ($env:CARD_GST) { Remove-Item Env:CARD_GST -ErrorAction SilentlyContinue }
    if (-not $env:CARD_HUMAN_NUM) { $env:CARD_HUMAN_NUM = "10" }
    if (-not $Smoke) {
        $sacHumans = if ($env:CARD_HUMAN_NUM -match '^\d+$') { [int]$env:CARD_HUMAN_NUM } else { 10 }
        if ($sacHumans -ge 15) {
            if (-not $PSBoundParameters.ContainsKey("TrainMaxSteps")) { $TrainMaxSteps = 250000 }
            if (-not $PSBoundParameters.ContainsKey("FinalistSteps")) { $FinalistSteps = 800000 }
        } else {
            if (-not $PSBoundParameters.ContainsKey("TrainMaxSteps")) { $TrainMaxSteps = 150000 }
            if (-not $PSBoundParameters.ContainsKey("FinalistSteps")) { $FinalistSteps = 550000 }
        }
        if (-not $PSBoundParameters.ContainsKey("NEvalEpisodes")) { $NEvalEpisodes = 10 }
        if (-not $PSBoundParameters.ContainsKey("EvalFreq")) { $EvalFreq = 10000 }
        if (-not $PSBoundParameters.ContainsKey("FinalEvalEpisodes")) { $FinalEvalEpisodes = 100 }
    }
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
$gstFlag = if ($env:CARD_GST -and $env:CARD_GST -match '^(1|true|yes|on)$') { "GST=on(inferred)" } else { "GST=off" }
$humanNum = if ($env:CARD_HUMAN_NUM -and $env:CARD_HUMAN_NUM -match '^\d+$') { [int]$env:CARD_HUMAN_NUM } else { 20 }
if ($Algo -eq "sac") {
    Write-Host "Stack: SAC + flat obs | env H=$humanNum robot_visible=False (no GST)"
} else {
    Write-Host "Stack: CrowdNav++ PPO/selfAttn SRNN + $gstFlag | env H=$humanNum robot_visible=False"
}
Write-Host "Device: CARD_DEVICE=$($env:CARD_DEVICE) torch_threads=4"
Write-Host "Features: CardPlus/SH=$CardPlus DatasetGate=$(-not $NoDatasetRescore) Finalists=$(-not $NoFinalists)"
Write-Host "Budget: short=$TrainMaxSteps x$TrainRounds finalists=$FinalistTopK x$FinalistSteps"
Write-Host "Eval: freq=$EvalFreq n_mid=$NEvalEpisodes held_out/final=$FinalEvalEpisodes"

$common = @(
    "--exp_name", "card",
    "--task_name", "circle-crossing-v0",
    "--algo", "$Algo",
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
# Two permanent configs: H=10 -> crowdnav_ds_h10, H=20 -> crowdnav_ds_h20 (legacy alias: crowdnav_ds_v1).
if (-not $NoDatasetRescore) {
    if ($DatasetPath -eq "") {
        $dsNamed = Join-Path $repo ("results\datasets\crowdnav_ds_h{0}" -f $humanNum)
        $dsLegacy = Join-Path $repo "results\datasets\crowdnav_ds_v1"
        if (Test-Path -LiteralPath $dsNamed) {
            $DatasetPath = $dsNamed
        } elseif ($humanNum -eq 20 -and (Test-Path -LiteralPath $dsLegacy)) {
            $DatasetPath = $dsLegacy
        } else {
            $DatasetPath = $dsNamed
        }
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
        if ($UseG2Judge -and $LearnedJudge -eq "") {
            $LearnedJudge = Join-Path $repo "results\learned_judge_g2.json"
        }
        if ($LearnedJudge -ne "") {
            if (-not (Test-Path -LiteralPath $LearnedJudge)) {
                throw "Learned judge not found: $LearnedJudge (run scripts/fit_learned_judge.py first)"
            }
            $common += @("--dataset_learned_judge", "$LearnedJudge")
            Write-Host "G2 learned judge ON: $LearnedJudge"
        }
        Write-Host "Dataset gate ON ($DatasetMode): $DatasetPath (LLM $LlmCandidates candidates -> best composite out)"
    } else {
        $msg = "Dataset not found at $DatasetPath. Collect first: python -m card_plus.dataset.collect --out results/datasets/crowdnav_ds_h$humanNum --human-num $humanNum --pool 1000 --n-cases 80 --min-near-path 0 --min-clearance 0.35 --selection balanced --randomization with_random --policies go_to_goal,noisy_go_to_goal,flee,stationary,random --random-seeds 0,1,2 --noise-std 0.3"
        if ($AllowMissingDataset) {
            Write-Host "WARNING: $msg - gate disabled (-AllowMissingDataset)"
        } else {
            throw $msg
        }
    }
}
Write-Host "TrainNum=$TrainNum EvalNum=$EvalNum VecEnv=$VecEnv CardPlus=$CardPlus"
$continueSteps = [Math]::Max(0, $FinalistSteps - $TrainMaxSteps)
$approxShort = $TrainRounds * $TrainMaxSteps
$approxFinal = $FinalistTopK * $continueSteps
Write-Host "LEAN budget: short=${TrainRounds}x${TrainMaxSteps}=$approxShort; finalists=${FinalistTopK}x+${continueSteps}=$approxFinal; total_RL~=$($approxShort+$approxFinal)"
Write-Host "eval: mid_train every ${EvalFreq} steps x ${NEvalEpisodes}eps (SH); end_of_short=${FinalEvalEpisodes}eps; llm_cand=$LlmCandidates; NoFinalists=$NoFinalists"

function Invoke-Card([string[]]$arguments, [string]$name) {
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

Write-Host "Starting CARD CrowdNav++ circle-crossing-v0 run"
Write-Host "cwd=$here TrainRounds=$TrainRounds TrainMaxSteps=$TrainMaxSteps Smoke=$Smoke"
Write-Host "XAI_KEYS_PATH=$($env:XAI_KEYS_PATH)"
Write-Host "XAI_PROXY=$($env:XAI_PROXY)"

$runsRoot = Join-Path $repo "results\runs\crowdnav"
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null

if ($ResumeLogPath -ne "") {
    $runLogPath = $ResumeLogPath
    Write-Host "Resuming LOG_PATH=$runLogPath from step $StartStep"
} else {
    $stamp = Get-Date -Format "yyyy-MM-dd-HH-mm-ss"
    if ($Algo -eq "sac") {
        $runLogPath = Join-Path $runsRoot ("SAC_H{0}_{1}" -f $humanNum, $stamp)
    } else {
        $runLogPath = Join-Path $runsRoot ("PPO_GST_H{0}_{1}" -f $humanNum, $stamp)
    }
    New-Item -ItemType Directory -Force -Path $runLogPath | Out-Null
    $StartStep = 0
    Invoke-Card (@("crowdnav_exp_one_step.py") + $common + @("--step", "0", "--log_path", $runLogPath)) "step0"
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
        }
    }
    if ($StartStep -le $k) {
        $stepArgs = @("crowdnav_exp_one_step.py") + $common + @("--step", "$k", "--log_path", $runLogPath)
        # If train already finished and checkpoint exists, skip re-training.
        $ckpt = Join-Path $runLogPath "checkpoint_$k.pkl"
        $tb = Join-Path $runLogPath "code_$($k - 1)\seed_1\tensorboard_log"
        $modelPt = Join-Path $tb "latest_model.pt"
        $modelZip = Join-Path $tb "latest_model.zip"
        $evalNpz = Join-Path $tb "eval\evaluations.npz"
        $finalEval = Join-Path $tb "eval\final_eval.json"
        $modelPath = if ($Algo -eq "sac") { $modelZip } else { $modelPt }
        $skippedTrain = $false
        if (Test-Path -LiteralPath $ckpt) {
            $stepArgs += "--checkpoint"
            $skippedTrain = $true
            Write-Host "Found $ckpt -> loading with --checkpoint (skip train)"
        } elseif ((Test-Path -LiteralPath $modelPath) -and ((Test-Path -LiteralPath $evalNpz) -or (Test-Path -LiteralPath $finalEval))) {
            # Train finished (eval artifacts present): skip retrain, finish collect/feedback.
            $stepArgs += "--load_model"
            Write-Host "Found completed $modelPath + eval -> --load_model (skip train)"
        } elseif (Test-Path -LiteralPath $modelPath) {
            # Mid-train interrupt: do NOT --load_model; Python resumes/heals.
            Write-Host "Found incomplete $modelPath (no eval yet) -> resume/retrain in Python"
        }
        Invoke-Card $stepArgs "step$k"
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
                "--algo", "$Algo",
                "--top_k", "$FinalistTopK",
                "--train_max_steps", "$FinalistSteps",
                "--short_train_steps", "$TrainMaxSteps",
                "--final_eval_seed", "1001",
                "--final_eval_episodes", "$FinalEvalEpisodes",
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

Write-Host "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "log: $runLogPath"
Write-Host "Resume:"
Write-Host ("  .\run_crowdnav_card.ps1 -Algo $Algo -ResumeLogPath LOG_PATH -StartStep N")
$resultsHint = Join-Path $repo "results\crowdnav"
if (Test-Path -LiteralPath $resultsHint) {
    $latestResult = Get-ChildItem -LiteralPath $resultsHint -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestResult) {
        Write-Host "curated results: $($latestResult.FullName)"
    }
}
