# Integration report: reward_triage ↔ CARD+

**Date:** 2026-11-14  
**CARD+:** `D:\Thesis\Implementation\card-v1.0`  
**Triage:** `D:\Thesis\Implementation\reward-surrogate-research`  
**Audit source:** triage repo `docs/CARD_AUDIT.md`

## 1. CARD architecture map and seams chosen

Process-per-step loop (shell): generate (LLM) → validate (`env_is_valid`) → TPE v1/v2 → SAC `train_and_eval` → feedback → next query.

CARD+ (flags default False): TPE v2, sub-reward stats, archive/rollback via `card_plus.hooks.apply_card_plus_after_baseline`.

| Seam | Site | Triage hook |
|------|------|-------------|
| After generation, before training | `metaworld_exp_one_step.py` / `crowdnav_exp_one_step.py` (step≥1, non-checkpoint) | `card_triage_adapter.before_train` — validate + cache lookup; **TPE still runs** |
| After each iteration | After `apply_card_plus_after_baseline` (or baseline feedback) | `after_iteration` — cache put, shadow log, plateau update |
| Termination | `triage_stop` in `rl_query_code_*.pkl` + `triage_plateau_stop.json` | Outer runner may break early |

Thin adapter only: repo-root `card_triage_adapter.py`. No triage code copied into CARD+.

## 2. What was simplified in reward-surrogate-research

- README “Start here” with four tools + honest status.
- Gate/phase/prereg docs → `docs/archive/`; runners → `scripts/research/` (+ `docs/archive/INDEX.md`).
- Core kept: `reward_triage/{schema,executor,tier0,label_cache,plateau}.py`, adapters, `scripts/mini_host_loop.py`.
- One product config: `configs/triage.yaml`.
- `pip install -e .` importable without torch.

## 3. Flags, defaults, config, how to run

| Flag | Default | Effect |
|------|---------|--------|
| `--triage-validate` | False | Tier-0 on short transitions; hard reject → skip RL |
| `--triage-cache` | False | Cache key = hash(normalized code, env, train, seeds); hit → skip RL |
| `--triage-plateau-stop` | False | Stop after `patience` non-improving iters (margin×SE); after rollback attempt if archive on |
| `--triage-shadow` | False | Log shadow score only; never filter |

```bash
# In CARD env
pip install -e /path/to/reward-surrogate-research

python metaworld_exp_one_step.py ... --triage-validate --triage-cache --triage-plateau-stop --triage-shadow
# Same flags on crowdnav_exp_one_step.py
```

Optional: `--triage-plateau-margin`, `--triage-plateau-patience`, `--triage-cache-root`.

Logs: `<log_path>/triage_log.jsonl`, `triage_label_cache/`, `triage_plateau_state.json`.

## 4. Cost accounting (smoke)

**Formula (per skipped train round):**  
`steps_saved_est = train_max_steps × train_seed_num`  
(Meta-World/CrowdNav use **SAC**, not PPO; same step budget accounting.)

| Tool (smoke test) | Measured vs estimated |
|-------------------|------------------------|
| Validate reject of constant reward | **Measured** skip; **estimated** steps = configured `train_max_steps` (no SAC run) |
| Cache hit after one put | **Measured** skip; steps from `train_steps_saved` in cache file |
| Plateau stop | **Measured** stop flag; remaining rounds × steps = **estimated** (runner must honor `triage_stop`) |
| Shadow | **0** steps saved (log only) |

Smoke (`tests/test_card_triage_adapter.py::test_smoke_all_flags_on`): no real SAC; savings are accounting of skips against the configured budget, not wall-clock GPU hours.

## 5. Test results

| Suite | Result |
|-------|--------|
| CARD+ `tests -k "card_plus or card_triage"` | **36 passed** |
| Golden (`test_card_plus_golden` / flags-off adapter no-op) | **pass** — baseline feedback + always-train unchanged when triage flags False |
| Per-flag unit tests (validate, cache, plateau, shadow) | **pass** |
| Smoke all flags on | **pass** |
| reward-surrogate-research `pytest -m "not slow"` | **pass** |

## 6. Risks and what was NOT done

- Surrogate **not** wired for filtering (shadow only).
- Real-LLM pilot **not** run; no API keys used.
- No claims about savings beyond the smoke / unit accounting above.
- Stochastic SAC + shared `compute_dense_reward` monkeypatch: cache is reuse of a prior label, not a formal identity.
- Plateau + archive: stop waits until a rollback attempt has been recorded when `--use_archive_rollback` is on; if rollback never fires, plateau may not stop.
- ManiSkill one-step **not** wired.
- Shell runners do not yet auto-break on `triage_stop` (flag is written for the runner to check).

## 7. Suggested next experiment

Real-LLM A/B on one CARD task (e.g. drawer-open-v2):

- Arm A: train-all (flags off).  
- Arm B: `--triage-validate --triage-cache --triage-plateau-stop` (optional `--triage-shadow`).  
- Several repeats; compare **total SAC steps to target** and **independently re-evaluated final success** (`card_plus.eval_best`).  
- Pre-register budgets and stop rules; do not tune after seeing results.
