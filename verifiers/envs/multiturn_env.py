from abc import abstractmethod
from copy import deepcopy
from typing import Tuple
import logging
import random

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from verifiers.envs.environment import Environment
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Info,
    Messages,
    MessageType,
    SamplingArgs,
    State,
)

logger = logging.getLogger(__name__)


def log_random(msg):
    if random.random() < 0.01:  # log 1%
        logger.info(msg)


class MultiTurnEnv(Environment):
    def __init__(
        self,
        message_type: MessageType = "chat",
        max_turns: int = 10,
        max_seq_len: int = 4096,
        tokenizer_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.message_type = message_type
        self.max_seq_len = max_seq_len

        if tokenizer_id is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            logger.info(f"Initializing {tokenizer_id} tokenizer for Environment")
        else:
            self.tokenizer = None

    @abstractmethod
    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        pass

    @abstractmethod
    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        """
        Generate a response from the environment (messages, state).
        """
        pass

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info = {},
        sampling_args: SamplingArgs = {},
        **kwargs,
    ) -> Tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        is_completed = False
        state = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
        }
        if self.message_type == "chat":
            assert isinstance(prompt, list)
            completion = []
        else:
            assert isinstance(prompt, str)
            completion = ""
        rollout = deepcopy(prompt)
        while not is_completed:
            if self.is_completed(rollout, state, **kwargs):
                is_completed = True
                break
            response = await self.get_model_response(
                client=client,
                model=model,
                prompt=rollout,
                oai_tools=info.get("oai_tools", None),
                sampling_args=sampling_args,
                message_type=self.message_type,
            )
            if response.usage.total_tokens >= self.max_seq_len -1:
                if self.message_type == "chat":
                    response.choices[0].message.content = "[ERROR] max_tokens_reached"
                else:
                    response.choices[0].text = "[ERROR] max_tokens_reached"
            state["responses"].append(response)
            if self.message_type == "chat":
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                assert isinstance(response, ChatCompletion)
                response_text: str = response.choices[0].message.content or ""  # type: ignore
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text,
                }
                if response.choices[0].message.tool_calls:
                    response_message["tool_calls"] = response.choices[
                        0
                    ].message.tool_calls  # type: ignore
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                assert isinstance(response, Completion)
                response_text: str = response.choices[0].text or ""  # type: ignore
                rollout += response_text
                completion += response_text
            state["turn"] += 1
            if (
                self.is_completed(rollout, state, **kwargs)
                or state["turn"] >= self.max_turns
            ):
                is_completed = True
            elif response.usage.total_tokens >= self.max_seq_len:
                msg = (
                    f"Stop rollout because current tokens ({response.usage.total_tokens}) "
                    f"exceeds max_seq_len ({self.max_seq_len})"
                )
                logger.info(msg)
                log_random(rollout)
                is_completed = True
            else:
                env_msgs, state = self.env_response(rollout, state, **kwargs)

                # env response might be long, especially when there are parallel tool calls.
                # we will stop rollout in that case.
                if self.tokenizer is not None:
                    num_new_tokens = len(self.tokenizer.apply_chat_template(env_msgs))
                    if response.usage.total_tokens + num_new_tokens >= self.max_seq_len:
                        msg = (
                            f"Stop rollout because current tokens ({response.usage.total_tokens}) "
                            f"+ environment tokens ({num_new_tokens})"
                            f"exceeds max_seq_len ({self.max_seq_len})"
                        )
                        logger.info(msg)
                        log_random(rollout)
                        is_completed = True

                if self.message_type == "chat":
                    assert isinstance(env_msgs, list)
                    assert isinstance(rollout, list)
                    assert isinstance(completion, list)
                    rollout += env_msgs
                    completion += env_msgs
                else:
                    assert isinstance(env_msgs, str)
                    assert isinstance(rollout, str)
                    assert isinstance(completion, str)
                    rollout += env_msgs
                    completion += env_msgs

        return completion, state
