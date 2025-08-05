import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Tuple

from verifiers.rubrics.tool_rubric import ToolRubric
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Message, Messages, State, RewardFunc
from verifiers.parsers.xml_parser import (
    XMLParser,
)

logger = logging.getLogger(__name__)


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()

    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip()
                        for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(
        sig.return_annotation.__name__
        if sig.return_annotation != inspect.Parameter.empty
        else "any"
    )

    print(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name) + 1:].strip()

        args[name] = {
            "type": str(
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            ),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default

    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples,
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]

        desc.append("\nArguments:")
        for arg_name, arg_info in schema["args"].items():
            default = (
                f" (default: {arg_info['default']})" if "default" in arg_info else ""
            )
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")

        if schema["examples"]:
            desc.append("\nExamples:")
            for example in schema["examples"]:
                desc.append(f"  {example}")

        if schema["returns"]:
            desc.append(f"\nReturns: {schema['returns']}")

        descriptions.append("\n".join(desc))

    return "\n\n".join(descriptions)


class ToolEnv(MultiTurnEnv):
    def __init__(
        self,
        tools: List[Callable] = [],
        system_prompt: str = "",
        format_prompt: bool = True,
        parser: XMLParser = XMLParser(
            fields=["think", ("tool_call", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["tool_response"]),
        max_turns: int = 10,
        **kwargs,
    ):
        rubric = ToolRubric(tools=tools, parser=parser, env_parser=env_parser)
        self.tool_schemas = [
            infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}

        if format_prompt:
            tool_descriptions = format_tool_descriptions(self.tool_schemas)
            formatted_prompt = system_prompt.format(
                tool_descriptions=tool_descriptions)
        else:
            formatted_prompt = system_prompt
        super().__init__(
            system_prompt=formatted_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
        self.env_parser = env_parser

    def get_reward_funcs(self, **kwargs) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(
        self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs: Any
    ) -> bool:
        return self.parser.parse_answer(messages) is not None

    def call_tool(self, tool_json: str, max_chars: int = 8192, **kwargs) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return 'Error: Parse tool '+tool_json + ' failed. Tool command must be a JSON object, e.g. \'{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}\''

            tool_name = command.get("name")
            if not tool_name:
                return 'Error: Tool command must specify \'name\', e.g. \'{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}\''

            if tool_name not in self.tools:
                return (
                    f"Error: Unknown tool '{tool_name}. "
                    + 'Please format your tool call as \'{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}\''
                )

            # follow Qwen3 template
            tool_func = self.tools[tool_name]
            tool_args = command.get("arguments", {})
            if isinstance(tool_args, str):
                tool_schema = next(
                    (
                        schema["args"]
                        for schema in self.tool_schemas
                        if schema["name"] == tool_name
                    ),
                    None,
                )
                return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a string."

            # Call the tool function with arguments
            result = tool_func(**tool_args)
            if max_chars > 0 and len(str(result)) > max_chars:
                result = str(result)[:max_chars] + "..."
            return str(result)
        except Exception as e:
            return (
                f"Error: call tool {tool_json} with error: '{str(e)}'. "
                + 'Please format your tool call as \'{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}\''
            )

    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Message, State]:
        parsed = self.parser.parse_many(messages[-1]["content"])
        results = []
        # Check if we got a valid tool field (not just None from failed parsing)
        if hasattr(parsed, "tool_call") and parsed.tool_call is not None:
            for tool in parsed.tool_call:
                try:
                    result = self.call_tool(tool)
                    if len(result.strip()) > 0:
                        results.append(self.env_parser.format(
                            tool_response=result))

                    else:
                        results.append(
                            f"<tool_response>\nError: Tool {tool} execution returned empty output.\n</tool_response>")

                except Exception as e:
                    print(e, tool, type(tool))
                    results.append(
                        f"<tool_response>\nError: Tool command {tool} not found or invalid XML format. Please ensure correct formatting.\n</tool_response>")
            if len(results) == 0:
                return [{
                    "role": "user",
                    "content": "<tool_response>\nError: Tool command not found or invalid XML format. Please ensure correct formatting.\n</tool_response>\n",
                }], state

            results = "\n".join(results)

            return [{"role": "user", "content": results}], state
        return [{
            "role": "user",
            "content": "<tool_response>\nError: Tool command not found or invalid XML format. Please ensure correct formatting.</tool_response>",
        }], state
