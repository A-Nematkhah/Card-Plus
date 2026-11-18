import sys, os
from stable_baselines3.common.callbacks import BaseCallback
import random
import numpy as np
import torch
import re
import colorama
colorama.init()
SUB_REWARD_PREFIX="sub_reward/"
MEAN_SUFFIX="_mean"
STD_SUFFIX="_std"
INFO_PREFIX="convert_info/"
COLOR_ENABLED=False


def print_with_color(content: str, color: str = "GREEN") -> None:
    if COLOR_ENABLED:
        print(getattr(colorama.Fore, color) + content + colorama.Style.RESET_ALL)
    else:
        print(content)


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if using_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

class SubRewardLog(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path, log_file_name, n_eval_episodes, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.log_file_name = log_file_name
        self.n_eval_episodes = n_eval_episodes
        self.learning_result = {}
        # init learning_result
        obs = self.eval_env.reset()
        next_obs, reward, done, info = self.eval_env.step(self.eval_env.action_space.sample())
        for k, v in info.items():
            if k.startswith(SUB_REWARD_PREFIX):
                self.learning_result[k] = []

    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            for k in self.learning_result.keys():
                self.learning_result[k].append([])
            for i in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_data = {}
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, done, info = self.eval_env.step(action)
                    for k, v in info.items():
                        if k.startswith(SUB_REWARD_PREFIX):
                            if k not in episode_data.keys():
                                episode_data[k] = 0
                            episode_data[k] += v
                    obs = next_obs
                # LLM reward dicts can add keys mid-training (e.g. collision only on some steps).
                for k in set(episode_data) | set(self.learning_result):
                    if k not in self.learning_result:
                        n_slots = 1
                        n_done_this_eval = 0
                        if self.learning_result:
                            sample = next(iter(self.learning_result.values()))
                            n_slots = len(sample)
                            n_done_this_eval = len(sample[-1]) if sample else 0
                        # Pad prior evals, and missed episodes in the current eval.
                        self.learning_result[k] = [
                            [0.0 for _ in range(self.n_eval_episodes)] for _ in range(max(n_slots - 1, 0))
                        ] + [[0.0] * n_done_this_eval]
                    self.learning_result[k][-1].append(float(episode_data.get(k, 0.0)))
            for k, v in self.learning_result.items():
                if v[-1]:
                    self.logger.record(f"ep_mean_{k}", np.mean(v[-1]))
        return super()._on_step()

    def on_training_end(self) -> None:
        # Logging must never abort a finished training run.
        try:
            os.makedirs(self.log_path, exist_ok=True)
            file_path = os.path.join(self.log_path, self.log_file_name)
            n_evals = 0
            for v in self.learning_result.values():
                n_evals = max(n_evals, len(v))
            to_save = {}
            for k, series in self.learning_result.items():
                rows = []
                for slot in series:
                    try:
                        values = [float(x) for x in slot]
                    except (TypeError, ValueError):
                        values = []
                    if len(values) < self.n_eval_episodes:
                        values = values + [0.0] * (self.n_eval_episodes - len(values))
                    else:
                        values = values[: self.n_eval_episodes]
                    rows.append(values)
                while len(rows) < n_evals:
                    rows.append([0.0 for _ in range(self.n_eval_episodes)])
                to_save[k] = np.asarray(rows, dtype=np.float64)
            if to_save:
                np.savez(file_path, **to_save)
            if self.verbose > 0:
                print(f"Sub rewards saved to {file_path}")
        except Exception as exc:
            print_with_color(f"SubRewardLog save failed (training continues): {exc}", "YELLOW")

def get_learning_result_mean(learning_result_list: list, save_path: str):
    data_all_dict = {}
    for learning_result in learning_result_list:
        for key in learning_result.keys():
            if key not in data_all_dict:
                data_all_dict[key] = []
            if key == "timesteps":
                data_all_dict[key].append(learning_result[key])
            else:
                data_all_dict[key].append(np.mean(learning_result[key], axis=1, keepdims=True))
    learning_result_mean_dict = {}
    for key in data_all_dict:
        if key == "timesteps":
            learning_result_mean_dict[key] = data_all_dict[key][0]
        else:
            learning_result_mean_dict[key] = np.concatenate(data_all_dict[key], axis=1)
            learning_result_mean_dict[key+MEAN_SUFFIX] = np.mean(learning_result_mean_dict[key], axis=1)
            learning_result_mean_dict[key+STD_SUFFIX] = np.std(learning_result_mean_dict[key], axis=1)
    np.savez(save_path, **learning_result_mean_dict)
    return {key: value for key, value in learning_result_mean_dict.items() 
            if key == "timesteps" or MEAN_SUFFIX in key or STD_SUFFIX in key}

def get_learning_result_feedback_simple(learning_result: dict, log_point_num):
    timesteps = np.asarray(learning_result['timesteps']).reshape(-1)
    n = len(timesteps)
    if n == 0:
        return "No evaluation timesteps recorded.\n"

    def _mean_std(key, axis=1):
        if key not in learning_result:
            return np.zeros(n), np.zeros(n)
        arr = np.asarray(learning_result[key], dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # Align shorter sub-reward series (late-appearing keys) to timesteps.
        if arr.shape[0] < n:
            pad = np.zeros((n - arr.shape[0], arr.shape[1]), dtype=np.float64)
            arr = np.concatenate([pad, arr], axis=0)
        elif arr.shape[0] > n:
            arr = arr[-n:]
        return np.mean(arr, axis=axis), np.std(arr, axis=axis)

    results_mean, results_std = _mean_std('results')
    ep_lengths_mean, ep_lengths_std = _mean_std('ep_lengths')
    successes_mean, successes_std = _mean_std('successes')
    learning_result_feedback = "Evaluation is conducted at specific training intervals. The average results of the episodes are as follows:\n"
    if n < log_point_num:
        log_interval = 1
    else:
        log_interval = n // log_point_num
    step_list = list(range(n - 1, -1, -log_interval))
    if len(step_list) > log_point_num:
        step_list = step_list[:log_point_num]
    step_list.reverse()
    for i in step_list:
        learning_result_feedback += (f"When step is {int(timesteps[i])}, "
                    f"return is {results_mean[i]:.1f} (±{results_std[i]:.1f}), "
                    f"episode length is {ep_lengths_mean[i]:.1f} (±{ep_lengths_std[i]:.1f}), "
                    f"success rate is {successes_mean[i]*100:.1f}% (±{successes_std[i]*100:.1f}%), ")
        for key in learning_result.keys():
            if SUB_REWARD_PREFIX in key:
                core_key = key.replace(SUB_REWARD_PREFIX, "")
                value_mean, value_std = _mean_std(key)
                learning_result_feedback += (
                    f"cumulative {core_key} for the trajectory is {value_mean[i]:.1f} (±{value_std[i]:.1f}), "
                )
        learning_result_feedback += "\n"
    if learning_result_feedback[-3:] == ", \n":
        learning_result_feedback = learning_result_feedback[:-3] + ".\n"
    print("learning result (simple) feedback is:")
    print(learning_result_feedback)
    return learning_result_feedback


def _jsonable_scalar(value):
    """Convert CrowdNav/gym info values into pickle-safe Python/numpy scalars."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    # crowd_sim.envs.utils.info.* event objects (Nothing/Collision/Timeout/...)
    mod = getattr(type(value), "__module__", "") or ""
    if mod.startswith("crowd_sim"):
        out = {"_crowd_sim_event": type(value).__name__, "label": str(value)}
        if hasattr(value, "min_dist"):
            try:
                out["min_dist"] = float(value.min_dist)
            except Exception:
                pass
        return out
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return str(value)


def sanitize_info(info):
    """Drop non-pickle-safe CrowdNav objects from an env info dict."""
    if not isinstance(info, dict):
        return {"_raw": _jsonable_scalar(info)}
    out = {}
    for key, value in info.items():
        key_s = key if isinstance(key, str) else str(key)
        if isinstance(value, dict):
            out[key_s] = sanitize_info(value)
        else:
            out[key_s] = _jsonable_scalar(value)
    return out


def sanitize_trajectory(traj: dict) -> dict:
    """Return a trajectory copy safe to pickle without CrowdNav imports."""
    if not isinstance(traj, dict):
        return traj
    out = dict(traj)
    infos = out.get("infos")
    if infos is None:
        return out
    cleaned = []
    for info in list(infos):
        cleaned.append(sanitize_info(info))
    out["infos"] = np.array(cleaned, dtype=object)
    return out


def sanitize_trajectories(trajs):
    return [sanitize_trajectory(t) for t in (trajs or [])]


def collect_trajectories(model, env, mapping_dicts, trajectory_num=10, benchmark_name="metaworld"):
    trajectories = []
    reward_sums = []
    for episode in range(trajectory_num):
        # CrowdNav++ SRNN keeps RNN state across predict(); reset each episode.
        if hasattr(model, "reset_hidden"):
            model.reset_hidden()
        obs = env.reset()
        done = False
        episode_data = {'obs': [], 'actions': [], 'rewards': [], 'dones': [], 'infos': []}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            next_obs, reward, done, info = env.step(action)
            
            episode_data['obs'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['dones'].append(done)
            episode_data['infos'].append(sanitize_info(info) if isinstance(info, dict) else sanitize_info({"_raw": info}))
            if benchmark_name == "metaworld":
                for key, value in mapping_dicts.items():
                    if key not in episode_data.keys():
                        episode_data[key] = []
                    if key == "self.goal_position":
                        episode_data[key].append(info[value])
                    else:
                        episode_data[key].append(eval(value))
            elif benchmark_name == "crowdnav":
                # Dual-view / GST path may return Dict obs to the policy; reward
                # features stay on the engineered flat vector in info["flat_obs"].
                flat_obs = obs
                if isinstance(info, dict) and info.get("flat_obs") is not None:
                    flat_obs = info["flat_obs"]
                    episode_data["obs"][-1] = flat_obs
                for key, value in mapping_dicts.items():
                    if key not in episode_data.keys():
                        episode_data[key] = []
                    episode_data[key].append(eval(value, {"np": np, "obs": flat_obs}))
            elif benchmark_name == "maniskill":
                for key, value in info.items():
                    if key.startswith(INFO_PREFIX):
                        if key not in episode_data.keys():
                            episode_data[key] = []
                        episode_data[key].append(value)
            else:
                raise RuntimeError("benchmark name error!")
            # Sub-reward keys can appear mid-episode; keep every series aligned with obs length.
            t_now = len(episode_data["obs"])
            seen_sub = {k for k in episode_data if k.startswith(SUB_REWARD_PREFIX)}
            step_sub = {}
            for key, value in info.items():
                if key.startswith(SUB_REWARD_PREFIX):
                    step_sub[key] = float(value)
            for key in seen_sub | set(step_sub):
                if key not in episode_data:
                    episode_data[key] = [0.0] * (t_now - 1)
                elif len(episode_data[key]) < t_now - 1:
                    episode_data[key].extend([0.0] * (t_now - 1 - len(episode_data[key])))
                episode_data[key].append(step_sub.get(key, 0.0))
            obs = next_obs
        
        for key in episode_data.keys():
            episode_data[key] = np.array(episode_data[key], dtype=object) if key == "infos" else np.array(episode_data[key])
        reward_sums.append(np.sum(episode_data['rewards']))
        trajectories.append(episode_data)
    return trajectories

def convert_traj_to_str(trajectory_list: list, mapping_dicts, log_interval=100, benchmark_name="metaworld"):
    feedback_list = []
    for traj in trajectory_list:
        ep_len = len(traj['obs'])
        if ep_len == 0:
            feedback_list.append("Empty trajectory.\n")
            continue
        feedback_str = ""
        sequence = [0] + list(range(log_interval-1, ep_len, log_interval))
        if sequence[-1] != ep_len-1:
            sequence.append(ep_len-1)
        for i in sequence:
            if i >= ep_len:
                continue
            feedback_str += f"When step is {i+1}, "
            for key, value in traj.items():
                try:
                    n = len(value)
                except TypeError:
                    continue
                if i >= n:
                    continue
                if benchmark_name in ("metaworld", "crowdnav"):
                    if key in mapping_dicts or "reward" in key:
                        cell = value[i]
                        formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                        if key == "rewards":
                            formatted_key = "reward"
                        try:
                            if isinstance(cell, np.ndarray) and cell.ndim >= 1:
                                formatted_value = ", ".join(f"{float(v):.1f}" for v in np.asarray(cell).reshape(-1))
                                feedback_str += f"{formatted_key} is [{formatted_value}], "
                            else:
                                feedback_str += f"{formatted_key} is {float(cell):.1f}, "
                        except (TypeError, ValueError):
                            continue
                elif benchmark_name == "maniskill":
                    if key.startswith(INFO_PREFIX) or "reward" in key:
                        cell = value[i]
                        formatted_key = key.replace(SUB_REWARD_PREFIX, '')
                        if key == "rewards":
                            formatted_key = "reward"
                        try:
                            if isinstance(cell, np.ndarray) and cell.ndim >= 1:
                                formatted_value = ", ".join(f"{float(v):.1f}" for v in np.asarray(cell).reshape(-1))
                                feedback_str += f"{formatted_key} is [{formatted_value}], "
                            else:
                                feedback_str += f"{formatted_key} is {float(cell):.1f}, "
                        except (TypeError, ValueError):
                            continue
            feedback_str += "\n"
        last_info = {}
        try:
            last_info = traj["infos"][-1]
            if not isinstance(last_info, dict):
                last_info = {}
        except Exception:
            last_info = {}
        solved = bool(last_info.get("success", last_info.get("is_success", False)))
        collided = bool(last_info.get("collided", last_info.get("collision", False)))
        timed_out = bool(
            last_info.get("timeout", False)
            or last_info.get("TimeLimit.truncated", False)
        )
        end_step = int(sequence[-1]) + 1 if sequence else ep_len
        if solved:
            feedback_str += f"At step {end_step}, the agent solves the task, so the trajectory ends.\n"
        elif collided:
            feedback_str += (
                f"At step {end_step}, the agent collided and failed the task, "
                f"so the trajectory ends.\n"
            )
        elif timed_out:
            feedback_str += (
                f"At step {end_step}, the agent still has not solved the task and the maximum "
                f"trajectory length of {end_step} is reached, so the trajectory ends.\n"
            )
        else:
            feedback_str += (
                f"At step {end_step}, the agent failed to solve the task, "
                f"so the trajectory ends.\n"
            )
        if feedback_str[-3:] == ", \n":
            feedback_str = feedback_str[:-3] + ". \n"
        feedback_list.append(feedback_str)
    return feedback_list

def get_eval_feedback(trajectory_list: list, mapping_dicts, log_interval=100, benchmark_name="metaworld"):
    if not trajectory_list:
        msg = "After training, no evaluation trajectories were available.\n"
        print("eval feedback is:")
        print(msg)
        return msg
    sums = []
    for traj in trajectory_list:
        try:
            sums.append(float(np.sum(traj["rewards"])))
        except Exception:
            sums.append(float("-inf"))
    max_index = int(np.argmax(sums))
    min_index = int(np.argmin(sums))
    if len(trajectory_list) == 1 or max_index == min_index:
        eval_feedback = (
            "After training, we evaluated the model. Only one distinct evaluation "
            "trajectory was available; it is shown below:\n"
        )
        feedback_list = convert_traj_to_str(
            trajectory_list=[trajectory_list[max_index]],
            mapping_dicts=mapping_dicts,
            log_interval=log_interval,
            benchmark_name=benchmark_name,
        )
        eval_feedback += (
            f"trajectory with return {sums[max_index]:.1f} is:\n" + feedback_list[0]
        )
    else:
        special_trajectory_list = [trajectory_list[max_index], trajectory_list[min_index]]
        eval_feedback = "After training, we evaluated the model and the two trajectories with the highest and lowest return are as follows:\n"
        feedback_list = convert_traj_to_str(
            trajectory_list=special_trajectory_list,
            mapping_dicts=mapping_dicts,
            log_interval=log_interval,
            benchmark_name=benchmark_name,
        )
        eval_feedback += f"trajectories with the highest return {sums[max_index]:.1f} is:\n" + feedback_list[0] + "\n"
        eval_feedback += f"trajectories with the lowest return {sums[min_index]:.1f} is:\n" + feedback_list[1]
    print("eval feedback is:")
    print(eval_feedback)
    return eval_feedback

def ensure_crowdnav_method_signature(reward_function_code: str) -> str:
    """Make ``compute_dense_reward`` a bound-method style ``(self, ...)`` if needed.

    Soft LLM prompts sometimes omit ``self``; patching that function onto
    ContinuousTaskWrapper / DualViewRewardWrapper then shifts ``action``/``obs``
    and crashes with broadcast errors like shapes (2,) (0,).
    """
    if re.search(r"def\s+compute_dense_reward\s*\(\s*self\b", reward_function_code):
        return reward_function_code
    return re.sub(
        r"def\s+compute_dense_reward\s*\(",
        "def compute_dense_reward(self, ",
        reward_function_code,
        count=1,
    )


def convert_function_external(reward_function_code: str, benchmark_name="metaworld"):
    if benchmark_name == "metaworld":
        reward_function_code = re.sub(r'def compute_dense_reward\(self, ', 'def compute_dense_reward(', reward_function_code)
        reward_function_code = re.sub(r"action\s*:\s*np\.ndarray\s*,\s*obs\s*:\s*np\.ndarray\s*\)\s*->\s*Tuple\s*\[\s*float\s*,\s*Dict\s*\[\s*str\s*,\s*float\s*\]\s*\]\s*:", 'action: np.ndarray, obs: np.ndarray, goal_position) -> Tuple[float, Dict[str, float]]:', reward_function_code)
        reward_function_code = re.sub(r'self.env._get_pos_goal\(\)', 'goal_position', reward_function_code)
        return reward_function_code
    elif benchmark_name == "maniskill":
        reward_function_code = re.sub(r'def compute_dense_reward\(self, ', 'def compute_dense_reward(', reward_function_code)
        return reward_function_code
    elif benchmark_name == "crowdnav":
        # Drop bound-method ``self`` (allow newline after '('); expose ``pre_obs``.
        reward_function_code = re.sub(
            r"def\s+compute_dense_reward\s*\(\s*self\s*,\s*",
            "def compute_dense_reward(",
            reward_function_code,
            count=1,
        )
        reward_function_code = re.sub(
            r"def\s+compute_dense_reward\s*\(\s*self\s*\)",
            "def compute_dense_reward()",
            reward_function_code,
            count=1,
        )
        if re.search(r"def\s+compute_dense_reward\s*\([^)]*\bpre_obs\b", reward_function_code) is None:
            reward_function_code = re.sub(
                r"(def\s+compute_dense_reward\s*\([^)]*?)(\)\s*(?:->[^:]+)?:)",
                r"\1, pre_obs=None\2",
                reward_function_code,
                count=1,
            )
        reward_function_code = re.sub(
            r"getattr\(\s*self\s*,\s*[\'\"]pre_obs[\'\"]\s*(?:,\s*None)?\s*\)",
            "pre_obs",
            reward_function_code,
        )
        reward_function_code = reward_function_code.replace("self.pre_obs", "pre_obs")
        # First-step / missing pre_obs must not crash offline TPE / return-check.
        if "if pre_obs is None:" not in reward_function_code:
            reward_function_code = re.sub(
                r"(def\s+compute_dense_reward\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n)",
                r"\1    if pre_obs is None:\n        pre_obs = obs\n",
                reward_function_code,
                count=1,
            )
        return reward_function_code
    else:
        raise RuntimeError("benchmark name error!")


def trajectories_return_check(reward_function, trajectories, mapping_dicts, threshold, log_interval=100, benchmark_name="metaworld", env=None):
    success_trajs = {
        "pos": [],
        "traj": [],
        "return": [],
        "reward_per_step": [],
    }
    failure_trajs = {
        "pos": [],
        "traj": [],
        "return": [],
        "reward_per_step": [],
    }
    for traj_pos, traj in enumerate(trajectories):
        rewards = []
        for step in range(len(traj['obs'])):
            if benchmark_name == "metaworld":
                result = reward_function(action=traj['actions'][step], obs=traj['obs'][step], goal_position=traj['self.goal_position'][step])
            elif benchmark_name == "crowdnav":
                pre_obs = traj['obs'][step - 1] if step > 0 else traj['obs'][step]
                try:
                    result = reward_function(
                        action=traj['actions'][step],
                        obs=traj['obs'][step],
                        pre_obs=pre_obs,
                    )
                except TypeError:
                    holder = type("PreObsHolder", (), {})()
                    holder.pre_obs = pre_obs
                    result = reward_function(
                        holder, traj['actions'][step], traj['obs'][step]
                    )
            elif benchmark_name == "maniskill":
                # ManiSkill dense rewards typically need live env state. Reset at
                # the start of each trajectory so trajs do not contaminate each other.
                if env is None:
                    raise RuntimeError("maniskill return-check requires an env for per-traj replay")
                if step == 0:
                    env.reset()
                ac = traj["actions"][step]
                _ob, result, _done, _info = env.step(ac)
            else:
                raise RuntimeError("benchmark name error!")
            if isinstance(result, tuple):
                reward, reward_dict = result
            else:
                reward = result
            rewards.append(reward)
        total_reward = np.sum(rewards)
        last_info = traj["infos"][-1] if len(traj.get("infos", [])) else {}
        if not isinstance(last_info, dict):
            last_info = {}
        if bool(last_info.get("success", last_info.get("is_success", False))):
            success_trajs["pos"].append(traj_pos)
            success_trajs["traj"].append(traj)
            success_trajs["return"].append(total_reward)
            success_trajs["reward_per_step"].append(total_reward / len(traj['obs']))
        else:
            failure_trajs["pos"].append(traj_pos)
            failure_trajs["traj"].append(traj)
            failure_trajs["return"].append(total_reward)
            failure_trajs["reward_per_step"].append(total_reward / len(traj['obs']))

    n_succ = len(success_trajs["pos"])
    n_fail = len(failure_trajs["pos"])
    if n_succ == 0 or n_fail == 0:
        # Vacuous cases: all-success can pass (nothing contradicts ranking);
        # all-fail must NOT pass (there is no evidence the reward ranks success).
        if n_succ == 0:
            correct_count = 0
            correct_rate = 0.0
            pass_flag = False
            feedback = (
                f"all trajectories failed (success=0, fail={n_fail}); "
                f"return check cannot pass without successful trajectories."
            )
        else:
            correct_count = n_succ
            correct_rate = 1.0
            pass_flag = correct_rate >= threshold
            feedback = (
                f"all trajectories succeeded (success={n_succ}, fail=0); "
                f"return check passes vacuously."
            )
    else:
        failure_max_r = max(failure_trajs["reward_per_step"])
        failure_max_index = failure_trajs["reward_per_step"].index(failure_max_r)
        correct_count = 0
        for r in success_trajs["reward_per_step"]:
            if r > failure_max_r:
                correct_count += 1
        
        correct_rate = correct_count / len(success_trajs["pos"])
        error_count = len(success_trajs["pos"]) - correct_count
        pass_flag = correct_rate >= threshold
        if pass_flag:
            feedback = f"check pass! success traj num={len(success_trajs['pos'])}, fail traj num={len(failure_trajs['pos'])}, correct_count={correct_count}"
        else:
            feedback = (
                "We collected several trajectories where some agents successfully solved the task, and others failed.\n"
                f"The average reward per step on a successful trajectory is expected to be higher than that on a failed trajectory.\n"
                "Using the reward function you provided to calculate the average reward per step for each trajectory, "
                f"We found {len(success_trajs['pos'])} successful trajectories, {error_count} of which have a smaller average reward per step than some of the failed trajectories.\n"
                f"This means that the reward function you provided is {correct_rate*100:.2f}% accurate in distinguishing successful trajectories. "
                f"This indicates a flaw in the design of the reward function. You need to revise the reward function to ensure that the average reward per step for most successful trajectories is higher than that of the failed trajectories.\n"
                )
            success_min_r = min(success_trajs["reward_per_step"])
            success_min_index = success_trajs["reward_per_step"].index(success_min_r)
            feedback_list = convert_traj_to_str(trajectory_list=[success_trajs["traj"][success_min_index],
                                                                 failure_trajs["traj"][failure_max_index]],
                                                mapping_dicts=mapping_dicts,
                                                log_interval=log_interval,
                                                benchmark_name=benchmark_name,
                                                )
            feedback += f"For example, this is a trajectory where the agent successfully solved the task, with a return of {success_trajs['return'][success_min_index]:.1f}, a length of {len(success_trajs['traj'][success_min_index]['obs'])}, and an average reward per step of {success_min_r:.1f}:\n"
            feedback = feedback + feedback_list[0] + "\n"
            feedback += f"However, the following shows a trajectory where the agent failed to solve the task, with a return of {failure_trajs['return'][failure_max_index]:.1f}, a length of {len(failure_trajs['traj'][failure_max_index]['obs'])}, and an average reward per step of {failure_max_r:.1f}:\n"
            feedback = feedback + feedback_list[1] + "\n"

    return correct_count, len(success_trajs["pos"]), correct_rate, pass_flag, feedback

class Logger(object):
    def __init__(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        self.terminal = sys.stdout
        # utf-8: LLM rewards / stack traces often contain non-cp1252 chars on Windows
        self.log = open(filename, "a", encoding="utf-8", errors="replace")
        self.stderr = sys.stderr

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def write_error(self, message):
        self.stderr.write(message)
        self.log.write(message)

    def write_warning(self, message):
        warning_message = f"WARNING: {message}\n"
        self.terminal.write(warning_message)
        self.log.write(warning_message)