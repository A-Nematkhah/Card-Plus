# CARD+ Phase-0 Notes

## Iteration loop (Meta-World)

Outer orchestration is process-per-step (shell / `run_drawer_open_card.ps1`):

1. `metaworld_exp_one_step.py --step 0` writes `rl_query_code_0.pkl`
2. `query_llm_metaworld.py` → LLM Coder → `code_{k}/general_*.py`, `specific_*.py`, `llm_response_code_k.pkl`
3. `metaworld_exp_one_step.py --step k` (k≥1):
   - load `code_{k-1}` + message history + `trajectories_for_check_list`
   - optional TPE via `trajectories_return_check` (`utils.py`)
   - **always** calls `train_and_eval` in the stock code (even if TPE fails)
   - builds `feedback_str` (process + trajectory, or preference)
   - writes `rl_query_code_k.pkl` for the next LLM call

Chat history is appended inside `generation.py` (`ZeroShotGenerator.self_reflection`).

## Trajectory schema (`collect_trajectories`)

Per trajectory dict (numpy arrays unless noted):

| Key | Meaning |
|-----|---------|
| `obs` | per-step observation |
| `actions` | per-step action |
| `rewards` | scalar reward used in training (LLM dense reward) |
| `dones` | done flags |
| `infos` | object array of info dicts; success via `infos[-1]['success']` / `is_success` |
| mapping keys | e.g. `self.robot.ee_position`, `self.obj1.position`, `self.goal_position` |
| `sub_reward/<name>` | per-step component values (prefix `sub_reward/`) |

Offline re-score of a **new** reward on stored trajs is already done by `trajectories_return_check` (calls `reward_function` per step). CARD+ reuses the same pattern for TPE v2 / stats when components must be recomputed.

## TPE (v1)

- Location: `utils.trajectories_return_check`
- Threshold: CLI `--threshold` (default `0.8`)
- Metric: **undiscounted** average per-step return `sum(r)/T` (no `gamma^t` in this codebase)
- Pass: fraction of successful trajs whose mean-per-step exceeds the **worst** (max) failed mean-per-step ≥ δ

**Compromise:** paper text says skip RL on TPE fail; this repo still trains (`# 2.2 eval no matter pass_flag`). CARD+ keeps that baseline when all flags are False. Only `use_tpe_v2` + `tpe_v2_mode=gate` + `hacked` skips RL.

## Feedback strings

- Process: `get_learning_result_feedback_simple` (eval curve + sub-reward means)
- Trajectory: `get_eval_feedback` → `convert_traj_to_str`
- Preference: string built inside `trajectories_return_check` on failure
- Assembled in `metaworld_exp_one_step.py` into `feedback_str` → next LLM `inference_results`

## Meta-World observation layout (verified in vendored `metaworld/.../sawyer_xyz_env.py`)

`_get_curr_obs_combined_no_goal()` (18-D):

- `obs[0:3]` hand / end-effector position
- `obs[3]` gripper openness (normalized finger distance)
- `obs[4:7]` object position
- `obs[7:11]` object quaternion
- (padding / second object slots as needed)

Full V2 obs frame-stacks previous curr-obs and appends goal (`_get_pos_goal()`). With partial observability the goal slice in `obs` may be zeros; CARD stores goal separately as `self.goal_position` from `info` / env. Progress uses **object pos + stored goal**, never ground-truth reward.

## Assumptions

1. CARD+ hooks are wired into both `metaworld_exp_one_step.py` and `crowdnav_exp_one_step.py`. CrowdNav progress uses robot→goal distance (`obs[0:2]` vs `obs[4:6]`). ManiSkill is not wired yet.
2. TPE v2 “discounted” return matches **this repo’s** mean-per-step (undiscounted `sum(r)/T`), not a new γ-sum. The original TPE in this clone is also not pairwise: it is the fraction of successful trajs whose mean-per-step beats the **max** failed mean.
3. Archive/rollback spans process boundaries via JSON on disk + text injected into `feedback_str` (no extra LLM call beyond the scheduled reflection).
4. **Last step** is `--card_plus_max_step` (the real last train round of the runner: drawer default 3, CrowdNav `$TrainRounds`). It is **not** `reflection_num` (default 10). `score_final_version` trains the last version once with the same `train_and_eval` protocol **only if it has no score yet** (typically TPE-v2 gate skip), then re-picks best. Extra cost is printed and written to `card_plus_log.jsonl` (`extra_score_final_version`, `extra_train_cost`).
5. scipy is used for Spearman when available; numpy rank-correlation / Mann–Whitney AUC fallbacks are provided.
6. Sub-reward names for offline re-eval come from the reward dict returned by `compute_dense_reward`.
7. TPE v2 runs only when `--use_return_check` is also on (needs the same stored pool + candidate re-score path as v1).
8. Tiny non-CARD+ fix while wiring: `correct_all` is initialized to `0` so checkpoints work if `--use_return_check` is off (previously an unbound name).
9. **No-success pool:** `pass_v1 := (rho_full >= tpe_v2_tau_dense)`, then `hacked` is computed normally. Skip-RL is still **only** `tpe_v2_mode=gate` AND `hacked`. The override is logged (`no_success_pool`, `pass_v1_overridden`).
10. Feedback order when CARD+ is on: process/trajectory (or preference) → dense-shaping check (if hacked) → sub-reward stats → rollback notice. If a rollback notice is appended, the stats header is labeled `for reward version {current_idx}` (the failed version the stats describe).
11. Final evaluation of the winner is **not** the CARD loop. `python -m card_plus.eval_best --log_path RUN --benchmark metaworld|crowdnav` reads `card_plus_best/specific.py`. Selection seed is 1 (runners); eval seed defaults to **1001**. Runners: `-EvalBest` (off by default).
12. Line-diff sizes vs best and vs previous version are logged per iteration (`diff_lines_vs_best`, `diff_lines_vs_prev`).
13. Dense return subtracts sparse components from the stored/recomputed **total**; it does not re-sum remaining keys (so `total != sum(components)` is handled).
14. Oracle TPE script (`python -m card_plus.tpe_v2_oracle`) uses an obs-only clone of drawer-open shaping (tolerance margins 0.16 / 0.5), not a live MuJoCo `compute_reward` call, so it runs without launching the env.

## Compromises forced by existing code

- **Stock CARD always trains after a TPE v1 failure.** In `metaworld_exp_one_step.py` this is explicit (`# 2.2 eval no matter pass_flag is True or False`). The paper text says skip RL; this clone does not. CARD+ preserves that baseline whenever all `use_*` flags are False. The only skip is `use_tpe_v2` + `tpe_v2_mode=gate` + `hacked`.
- The “loop” is process-per-step; archive persistence + `card_plus_best/` replace an in-process `return best_code`.
- Rollback does not rewrite `specific_*.py` on disk; it injects a notice + best code into the next Coder `feedback_str`. The separate `eval_best` entry point is what actually *evaluates* the archived file.
- Golden regression vs git `6e1df1c` is implemented by `assemble_baseline_feedback` + `simulate_one_step` (same helpers both one-step scripts call). Importing the one-step modules in pytest would pull SAC/MuJoCo/CrowdNav.

## Phase 0 — measurement (Go)

- **Cost ledger:** `card_plus_cost.jsonl` via `card_plus.logging_utils.append_cost_event` (fields: `ts`, `stage`, `iteration`, `prompt_tokens`, `completion_tokens`, `llm_calls`, `rl_steps`, `wall_s`, `extra`). Written from `hooks.write_iteration_log`, `dataset/candidate_loop`, and dry-run.
- **Stats:** `card_plus.stats_utils` — Wilson CI, bootstrap mean/median, per-seed report.
- **Archive:** `VersionRecord` optional fields `gate_composite`, `gate_metrics`, `online_success` (backward-compatible).
- **Table:** `python scripts/collect_gate_table.py --roots results/runs/crowdnav --out results/gate_summary.json`
- **Security:** `xai_keys.json` remains in `.gitignore`.
- **Ablation naming:** thesis configs are `C0…C6` (not `A0…A6`) to avoid clash with brainstorm IDs (`A1` attribution).

## Phase 1 — gate feedback (Go / flags default off)

- `--dataset_gate_attribution`: failed-pair sub-reward attribution in `DatasetGateResult.feedback_text`.
- `--dataset_suicide_probe`: early-collision **sum** must be worse than long-wander sum; rewrite prompt adds collision > horizon × per-step rule.
- `--dataset_counterfactual`: writes `dataset_gate*_counterfactual.json` (drop / ×2 components).
- Gate metrics include `gap_*` (value − floor) for threshold miscalibration diagnosis.
- Candidate summary also logs `llm_calls_per_accepted`, `median_retries`, `gate_pass_rate`.
- **Exit criteria (revised):** features behind flags; golden unchanged with flags off; track cost metrics on real runs — do not require a hard 40% call cut without an LLM run.

## Phase 2 — calibration & dataset v2 (critical Go/No-Go)

- **Split (B3):** `scenario_split.json` via `card_plus.dataset.split` (stratified by `humans_near_path`). Gate flags: `--dataset_split {all,train,val}`, `--dataset_split_seed`, `--dataset_val_fraction`. Helper: `python scripts/write_scenario_split.py --dataset DIR`.
- **Dataset v2 (B1):** `DATASET_FORMAT_VERSION=2` in collect meta; policies include `slow_go_to_goal`, `cautious` (ORCA-like), optional `sac` checkpoint. Recipe in `card_plus/dataset/collect.py` docstring.
- **Reward zoo / calibration (B2):** `card_plus.dataset.reward_zoo` + `card_plus.dataset.calibration`. Entry point:
  `python scripts/run_phase2_calibration.py --dataset DIR --split val --harvest-runs RUN ... --out results/phase2_calibration.json`
  Reports Spearman/Kendall(composite, online_success), composite component ablation, difficulty-bucket pair metrics.
- **Go/No-Go:** Spearman ≥ ~0.5 with N ≥ 20 labeled online successes → continue. Else redesign weights / G2 before Phase 3.
- **Note:** Filling `online_success` in the zoo JSON still requires short SAC H10 runs (2 seeds) per reward; the tooling scores offline composite immediately and decides Go once labels exist.

## Phase 3 — structure vs coefficients

- **C1:** `card_plus.dataset.coeff_ast` extract/inject numeric literals; `card_plus.dataset.coeff_optimize` maximizes train composite with hard anti-hack probe constraint; validates val composite. Methods: `auto` (CMA if installed → scipy DE → random), `cma`, `de`, `random`.
- **Flags (default off):** `--dataset_optimize_coeffs`, `--dataset_coeff_method`, `--dataset_coeff_max_evals`, `--dataset_structure_prompt` (also auto-enabled when optimize is on in candidate prompts).
- **Wiring:** after a candidate passes the gate, coeffs are tuned on `split=train`, re-gated, report → `coeff_optimize_{step}_{suffix}.json`.
- **C2/C3:** `card_plus.dataset.reward_shaping` — `RunningScaleNormalizer`, `potential_based_progress_value`, `STRUCTURE_ONLY_PROMPT`, helper snippet for potential-based progress.
- **CLI:** `python scripts/optimize_reward_coeffs.py --reward specific.py --dataset DIR --method random --out results/coeff_optimize.json`
- **Exit:** meaningful train/val composite gain without val drop (`CoeffOptimizeResult.ok`); online success non-regression checked later in Phase-6 ablations (`C4` = +coeff optimize).

## Phase 2 close-out (tooling + current evidence)

- Zoo build/label: `scripts/label_reward_zoo_online.py` (`--from-archives` / `--import-archives` / continuous proxies; optional `--train`).
- Calibration artifact: `results/phase2_calibration.json` — continuous `online_proxy` (`mean_progress`), **N=7**, Spearman≈**-0.92** → **No-Go** (decision: `results/phase2_decision.json`).
- Diagnosis: `intra_scenario` strongly anti-correlated (ρ≈−0.96); `anti_hack` only positive core signal (ρ≈+0.52). Positive-only weight grid still negative (best≈−0.63).
- Score redesign v2 defaults: `w_rank=0.05, w_sep=0.15, w_hack=0.65, w_intra=0.15` in `calibrated_composite`.
- **G2:** `card_plus.dataset.learned_judge` + `scripts/fit_learned_judge.py` → `results/learned_judge_g2.json` (signed weights; report train + LOO Spearman). Gate flag: `--dataset_learned_judge PATH` (default off; composite = `gate_score` ∈(0,1); absolute `min_composite` skipped under G2 — pair/anti_hack floors + max-composite selection remain). Phase 3 stays locked until honest Go.
- **Hard rule:** Phase 4–6 flags may ship defaults-off; **thesis Phase-2 Go** needs Spearman≥0.5 with honest validation (prefer LOO / held-out), not all-zero short SAC.

## Phase 4 — screen / ASHA / seed-aware rollback

- `card_plus/dataset/screen_rank.py` + `scripts/eval_screen_predictors.py` (current archive scan: best=success_at_1 Spearman≈0.87, `go=true` vs baseline).
- ASHA multi-seed: `successive_halving.asha_rung` / `asha_bracket` (median across seeds).
- Flags: `--use_asha_screen`, `--asha_n_seeds`, `--rollback_std_k` (`VersionRecord.score_std` / `n_seeds`).

## Phase 5 — Coder feedback

- `card_plus/hypothesis_loop.py` + `--use_hypothesis_loop`
- `card_plus/failure_feedback.py` + `--use_failure_cluster_feedback`
- `--dataset_max_diff_gate` / `--dataset_max_diff_lines` in candidate loop
- Coder model ablation harness: `scripts/ablate_coder_models.py`

## Phase 6 — thesis matrix

- `scripts/run_thesis_ablation.py` → `results/thesis_ablation_matrix.json` (C0–C6 flag sets)
- `scripts/summarize_thesis_tables.py` → Wilson/bootstrap cost×success tables
- Protocol locked: ≥3 train seeds, eval seed 1001, 100–200 eps

## H2-lite (parallel)

- `card_plus/llm_cache.py`: prompt/response disk cache + `llm_prompt_response.jsonl` audit
- H1 Python orchestrator deferred until after Phase-6 tables

## Stretch (Phase 7)

- G2 **activated** after Phase-2 No-Go (continuous-proxy evidence); expand labels when possible. G5 transfer later; skip G1/G4 unless requested.

