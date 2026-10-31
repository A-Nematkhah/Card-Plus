import re
import time
from typing import Any, List, Mapping, Optional, Tuple
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer, pipeline
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
try:
    from langchain.schema import AIMessage, BaseMessage
except ImportError:
    from langchain_core.messages import AIMessage, BaseMessage
from code_generation.post_process.post_process import RewardFunctionConverter
from code_generation.self_reflection.utils import print_with_color
from key_manager import get_xai_key_manager, is_xai_model

AZURE_MODELS = ["gpt-3.5-turbo", "gpt-35-turbo-1106", "gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-1106-preview", "gemini-1.5-pro-preview-0409"]
XAI_MAX_TOKENS = 16384
DEFAULT_XAI_REASONING_EFFORT = "high"
_CHAT_ROLES = {"system": "system", "human": "user", "ai": "assistant"}


def _strip_think_tags(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()


class XAIChatModel:
    """LangChain-style chat model backed by xAI (Grok)."""

    def __init__(self, model_name: str, temperature: float, reasoning_effort: Optional[str] = DEFAULT_XAI_REASONING_EFFORT) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.key_manager = get_xai_key_manager()

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        kwargs: dict = dict(
            model=self.model_name,
            messages=[{"role": _CHAT_ROLES[message.type], "content": message.content} for message in messages],
            temperature=self.temperature,
            max_completion_tokens=XAI_MAX_TOKENS,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.key_manager.chat_completion(**kwargs)
        return AIMessage(content=_strip_think_tags(response.choices[0].message.content))


class ZeroShotGenerator:
    def __init__(self, log_path: str, model_name="gpt-4", temperature=0.7, benchmark_name="metaworld",
                 validation_function=None,
                 max_try_num: int = 3,
                 args=None,
                 **kwargs) -> None:
        self.temperature = temperature
        if is_xai_model(model_name):
            reasoning_effort = getattr(args, "reasoning_effort", None) if args is not None else None
            if reasoning_effort is None:
                reasoning_effort = DEFAULT_XAI_REASONING_EFFORT
            self.llm = XAIChatModel(
                model_name=model_name,
                temperature=self.temperature,
                reasoning_effort=reasoning_effort,
            )
        elif model_name in AZURE_MODELS:
            self.llm = AzureChatOpenAI(
                deployment_name=model_name,
                openai_api_type="azure",
                temperature=self.temperature,
                **kwargs)
        else:
            raise ValueError(
                f"Model name {model_name} not supported! "
                "Use a grok-* / xai/* model (needs XAI_API_KEY) or an Azure model."
            )
        self.benchmark_name = benchmark_name
        self.log_path = log_path
        self.validation_function = validation_function
        self.max_try_num = max_try_num
        self.args = args
        self.model_name = model_name
        self.message_history = []

    def generate_code(self, chat_message, map_dict: dict, step, generate_type):
        try_num = 0
        converter = RewardFunctionConverter(map_dict)

        while try_num + 1 <= self.max_try_num:
            try_num += 1
            success_flag = False
            # AIMessage
            response = self.llm.invoke(chat_message)
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            # get code content
            for pattern in patterns:
                match = re.search(pattern, response.content, re.DOTALL)
                if match:
                    break
            if match:
                general_code = match.group(1)
                # Post-processing, replace the general terms with specific terms
                specific_code = converter.general_to_specific(general_code)

                is_valid, check_msg = self.validation_function(specific_code, self.args)
                if is_valid:
                    success_flag = True
                    break
                else:
                    print(check_msg)
                    continue
            else:
                time.sleep(10)
                continue

        if success_flag:
            return response, general_code, specific_code
        else:
            print(f"Maximum number of tries ({self.max_try_num}) reached. To prevent API token waste, terminate the program.")
            exit()

    def save_code(self, general_code, specific_code, step):
        code_dir = os.path.join(self.log_path, f"code_{step}")
        os.makedirs(code_dir, exist_ok=True)
        general_code_path = os.path.join(code_dir, f"general_{step}.py")
        specific_code_path = os.path.join(code_dir, f"specific_{step}.py")
        # LLM code may include unicode (e.g. →); Windows default cp1252 would crash.
        with open(general_code_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(general_code)
        with open(specific_code_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(specific_code)

        print_with_color(content=f"{step}_general code save to {general_code_path}", color="GREEN")
        print_with_color(content=f"{step}_specific code save to {specific_code_path}", color="GREEN")
        return general_code_path, specific_code_path

    def first_generation(self, system_prompt: str, first_generation_prompt: str, instruction: str, map_dict: dict, step=0) -> Tuple[str, str]:
        # add prompt to message history
        system_message = SystemMessagePromptTemplate.from_template(system_prompt).format()
        first_generation_message = HumanMessagePromptTemplate.from_template(first_generation_prompt).format(instruction=instruction)
        self.message_history.extend([system_message, first_generation_message])

        chat_message = ChatPromptTemplate.from_messages(self.message_history).format_prompt().to_messages()
        response, general_code, specific_code = self.generate_code(chat_message=chat_message, map_dict=map_dict,
                                                         step=step, generate_type="first_generation")
        # add response to message history
        self.message_history.append(response)
        # save reward function
        general_code_path, specific_code_path = self.save_code(general_code, specific_code, step)
        return general_code, specific_code, general_code_path, specific_code_path

    def self_reflection(self, self_reflection_prompt: str, instruction: str, map_dict: dict, inference_results: str, step: int):
        # add prompt to message history
        self_reflection_message = HumanMessagePromptTemplate.from_template(self_reflection_prompt).format(inference_results=inference_results)
        self.message_history.append(self_reflection_message)
        chat_message = ChatPromptTemplate.from_messages(self.message_history).format_prompt().to_messages()
        response, general_code, specific_code = self.generate_code(chat_message=chat_message, map_dict=map_dict,
                                                    step=step, generate_type="self_reflection")

        # add response to message history
        self.message_history.append(response)
        # save reward function
        general_code_path, specific_code_path = self.save_code(general_code, specific_code, step)
        return general_code, specific_code, general_code_path, specific_code_path