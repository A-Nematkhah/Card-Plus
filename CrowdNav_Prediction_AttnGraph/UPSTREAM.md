# Vendored: CrowdNav++

| Field | Value |
|-------|-------|
| Upstream | https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph |
| Used by | CARD CrowdNav benchmark in `code_generation/self_reflection/` |
| Simulator | `crowd_sim.envs.crowd_sim_var_num.CrowdSimVarNum` |
| Humans | ORCA via Python-RVO2 (`third_party/Python-RVO2`) |

CARD does **not** import upstream `crowd_nav.configs.config.Config` (it calls `get_args()` at import).
Instead `crowdnav_env_utils.build_config` builds a compatible namespace with `sim.predict_method = "none"`.
