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

