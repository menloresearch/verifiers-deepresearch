import json
import random
import re
import string
from typing import Callable, Dict, List

from verifiers.parsers import Parser, XMLParser
from verifiers.rubrics import Rubric


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


class ToolRubric(Rubric):
    def __init__(self,
                 parser: Parser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: Parser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        super().__init__(parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {
            tool.__name__ if hasattr(tool, '__name__') else str(tool): tool
            for tool in tools
        }
        self.reward_funcs = [
            self.correct_answer_reward_func,
            self.tool_execution_reward_func,    
            self.parser.get_format_reward_func(),
            self.efficient_thinking_reward_func,
            self.num_xml_reward_func
        ]
        self.reward_weights = [
            1.0,
            0.2,
            0.2,
            0.1,
            0.1
        ]
        # fixme: harcoded for current strat. maybe you want to do for tool_name in self.tools.keys(): ...
        # Tool execution success reward
        # for tool_name in self.tools.keys():
        #     self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
        #     self.reward_weights.append(0.0)
        #     self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
        #     self.reward_weights.append(0.0)
        #     self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
        #     self.reward_weights.append(0.0)
        # for tool_name in self.tools.keys():
        #     self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
        #     # Higher weight for search and visit tools
        #     if tool_name in ["web_search", "visit_tool"]:
        #         self.reward_weights.append(0.15)  # Reward successful search/visit
        #     else:
        #         self.reward_weights.append(0.0)

        #     # Tool usage count reward
        #     self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
        #     if tool_name in ["web_search", "visit_tool"]:
        #         self.reward_weights.append(0.15)  # Encourage using search/visit appropriately
        #     else:
        #         self.reward_weights.append(0.0)

            # # Tool attempt reward
            # self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            # if tool_name in ["web_search", "visit_tool"]:
            #     self.reward_weights.append(0.0)  # Small reward for attempting search/visit
            # else:
            #     self.reward_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import signal
        import sys
        from contextlib import redirect_stdout

        try:
            test_cases = json.loads(answer)["test_cases"]
        except:
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith("```python"):
            code_str = code_str[9:]
        elif code_str.startswith("```"):
            code_str = code_str[3:]
        if code_str.endswith("```"):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return "\n".join(line.strip() for line in output.splitlines())

        total_cases = 0
        passed = 0

        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test["input"])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test["output"])

                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1

            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__

        return passed / total_cases if total_cases else 0.0
 
    def correct_answer_reward_func(self, completion, answer, task, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        if task == "mc":
            response = str(self.parser.parse_answer(completion))
            reward = 1.0 if response == answer.strip() else 0.0
        elif task == "math":
            response = str(self.parser.parse_answer(completion))
            reward = 1.0 if answer == response else 0.0
        elif task == "code":
            response = str(self.parser.parse_answer(completion))
            reward = self.evaluate_code(response, answer, **kwargs)
        elif task == "qa":
            reward = self.qa_reward_func(completion, answer, task, **kwargs)
        else:
            reward = 0.0
        return reward
    
    def efficient_thinking_reward_func(self,prompt, completion, answer, task, **kwargs ) -> float:
        num_think_tag = 0
        total_token = 0
        for compl in completion:
            if compl.get("role") == "assistant":
                think_tags = re.findall(r'<think>(.*?)</think>', compl.get("content",""), re.DOTALL)
                
                for content in think_tags:
                    num_think_tag += 1
                    # Remove leading/trailing whitespace and split into tokens
                    tokens = content.strip().split()
                    total_token += len(tokens)
        return 100*num_think_tag/(1+total_token/(num_think_tag+0.001))
    
    def num_xml_reward_func(self,prompt, completion, answer, task, **kwargs ) -> float:
        import re

        def find_num_tags(text:str,tag :str) -> int:
            """
            Finds all content between <smt>...</smt> tag pairs in a text string.
            
            Args:
                text: Input string to search for tags
                
            Returns:
                List of strings containing all text between tag pairs
            """
            # pattern = rf'<{tag}>(.*?)</{tag}>'
            # pattern = rf'<{tag}>\s*(\{{.*?\}})\s*</{tag}>'
            pattern = rf'<{tag}>\s*(\{{\s*".*?"\s*:\s*.+?\s*}})\s*</{tag}>'
            return len(re.findall(pattern, text, flags=re.DOTALL))
        
        num_tool = 0
        num_think = 0
        num_answer = 0
        total_reward = 2.0
        for compl in completion:
            if compl.get("role") == "assistant":
                content = compl.get("content","")
                num_tool += find_num_tags(content, "tool")
                num_think += find_num_tags(content, "think")
                num_answer += find_num_tags(content, "answer")
                if content.count("<tool>") != content.count("</tool>") and (content.count("<tool>") > 0 or content.count("</tool>") >0):
                    total_reward -= 1.
                if content.count("<think>") != content.count("</think>") and (content.count("<think>") > 0 or content.count("</think>") >0):
                    total_reward -= 1.
                if content.count("<answer>") != content.count("</answer>") and (content.count("<answer>") > 0 or content.count("</answer>") >0):
                    total_reward -= 1.

                if find_num_tags(content, "tool") > 0 and find_num_tags(content, "answer") > 0 :
                    total_reward -= 2.
            
        return total_reward

    def qa_reward_func(self, completion, answer, task, **kwargs) -> float | None:
        """
        Reward function that checks if the QA answer matches the expected answer.
        Uses text normalization and either exact matching or substring matching.
        """
        match_mode = "substring"
        
        if task == "qa":
            response = str(self.parser.parse_answer(completion))

            # Try to parse the answer as a list of possible answers
            try:
                answers = json.loads(answer)
                if isinstance(answers, str):
                    answers = [answers]
                elif not isinstance(answers, list):
                    answers = [str(answers)]
            except json.JSONDecodeError:
                answers = [answer]

            # Get matching mode - default to exact match
            # match_mode = kwargs.get("qa_match_mode", "exact")
            if match_mode == "substring":
                reward = 1.0 if any(subem_check(response, ans) for ans in answers) else 0.0
            else:
                reward = 1.0 if any(em_check(response, ans) for ans in answers) else 0.0
        else:
            reward = None
        
        return reward

               

    def tool_execution_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        tool_attempts = 0
        successful_executions = 0
        
        # Find assistant messages with tools and their responses
        for i, msg in enumerate(completion):
            if msg['role'] == 'assistant':
                # Use parser to check for tool tag
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'tool') and parsed.tool is not None:
                    # Found a properly formatted tool message
                    if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                        tool_attempts += 1
                        # Check response with env_parser
                        parsed_response = self.env_parser.parse(completion[i + 1]['content'])
                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                            successful_executions += 1
        
        # Calculate reward
        if tool_attempts == 0:
            return 0.0
        return (successful_executions / tool_attempts)
    
    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """
        def tool_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            """
            Reward function that checks execution success for the {tool_name} tool.

            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json
            
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with the specific tool and their responses
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if isinstance(command, dict) and command.get("name") == tool_name:
                                # Found a properly formatted tool message for the specific tool
                                if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                                    tool_attempts += 1
                                    # Check response with env_parser
                                    parsed_response = self.env_parser.parse(completion[i + 1]['content'])
                                    if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)
             
        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func

    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_count_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            successful_executions = 0.0
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                                        parsed_response = self.env_parser.parse(completion[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
            return successful_executions
        
        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_attempt_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            attempted_executions = 0.0
            for i, msg in enumerate(completion):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if isinstance(command, dict) and command.get("name") == tool_name:
                                attempted_executions += 1
                        except json.JSONDecodeError:
                            pass
            return attempted_executions
            
        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func
