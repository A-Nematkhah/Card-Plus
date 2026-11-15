import argparse
import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generation import ZeroShotGenerator
from code_generation.self_reflection.benchmark_prompt.crowdnav_prompt import (
    crowdnav_first_generation_prompt,
    crowdnav_system_prompt,
)
from code_generation.self_reflection.benchmark_prompt.common_prompt import self_reflection_prompt
from code_generation.self_reflection.crowdnav_exp import instruction_mapping, mapping_dicts
from code_generation.self_reflection.crowdnav_exp_one_step import LLM_RESPONSE, env_is_valid
from code_generation.self_reflection.utils import Logger, print_with_color
from card_plus.dataset.candidate_loop import select_best_reward_candidates
from card_plus.dataset.gate import dataset_gate_from_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_question_file_path", type=str)
    args = parser.parse_args()

    with open(args.rl_question_file_path, "rb") as file:
        loaded_objects_dict = pickle.load(file)
        rl_args = loaded_objects_dict["rl_args"]
        step = loaded_objects_dict["step"]
        message_history = loaded_objects_dict["message_history"]
        trajectories_for_check_list = loaded_objects_dict["trajectories_for_check_list"]
        feedback_str = loaded_objects_dict["feedback_str"]

    real_log_path = os.path.dirname(args.rl_question_file_path)
    std_file = os.path.join(real_log_path, f"{LLM_RESPONSE}_{step}.log")
    logger = Logger(filename=std_file)
    sys.stdout = logger
    sys.stderr = logger
    print("args is:")
    print(json.dumps(vars(args), indent=4))

    code_generator = ZeroShotGenerator(
        log_path=real_log_path,
        model_name=rl_args.llm_model_name,
        temperature=rl_args.temperature,
        benchmark_name=rl_args.benchmark_name,
        validation_function=env_is_valid,
        max_try_num=rl_args.max_try_num,
        args=rl_args,
    )
    code_generator.message_history = message_history

    if step == 0:
        code_generator.seed_first_generation(
            system_prompt=crowdnav_system_prompt,
            first_generation_prompt=crowdnav_first_generation_prompt,
            instruction=instruction_mapping[rl_args.task_name],
        )
    else:
        code_generator.seed_self_reflection(
            self_reflection_prompt=self_reflection_prompt,
            inference_results=feedback_str,
        )

    gate_cfg = dataset_gate_from_args(rl_args)
    n_candidates = int(getattr(rl_args, "llm_candidates", 3))
    max_retries = int(getattr(rl_args, "llm_candidate_retries", 5))

    if gate_cfg.enabled and n_candidates >= 1:
        print_with_color(
            f"Multi-candidate LLM selection ON (n={n_candidates}, retries={max_retries})",
            "GREEN",
        )
        select_best_reward_candidates(
            code_generator,
            map_dict=mapping_dicts,
            step=step,
            gate_cfg=gate_cfg,
            n_candidates=n_candidates,
            max_retries_per_slot=max_retries,
            run_dir=real_log_path,
        )
    else:
        response, general_code, specific_code = code_generator.propose_reward(
            mapping_dicts, step, generate_type="first_generation" if step == 0 else "self_reflection"
        )
        code_generator.message_history.append(response)
        code_generator.save_code(general_code, specific_code, step)

    with open(os.path.join(real_log_path, f"{LLM_RESPONSE}_{step}.pkl"), "wb") as file:
        pickle.dump(
            {
                "trajectories_for_check_list": trajectories_for_check_list,
                "message_history": code_generator.message_history,
            },
            file,
        )


if __name__ == "__main__":
    main()
