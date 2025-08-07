import json
import re
import os
import string
import numpy as np
from typing import Callable, Dict, List
from scipy.stats import skewnorm
from verifiers.parsers.parser import Parser
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from verifiers.rubrics.utils.correctness_reward_utils import process_row


def find_fuzzy_substring(term: str, sentence: str, n: int = 1) -> tuple[bool, int, int]:
    """
    Checks if a term exists in a sentence allowing for up to n different characters.

    This function uses a dynamic programming approach based on the Levenshtein
    distance to find the best match of a 'term' within a 'sentence'.

    Args:
        term: The substring to search for.
        sentence: The text to search within.
        n: The maximum number of allowed differences (edits: insert, delete, substitute).

    Returns:
        A tuple containing:
        - bool: True if a match with at most n differences is found, False otherwise.
        - int: The minimum number of differences found for the best match.
        - int: The end position of the best match in the sentence. Returns -1 if no match.
    """
    # Get lengths of the term and sentence
    p = len(term)
    m = len(sentence)

    # --- 1. Create and Initialize the DP Table ---
    # Create a DP table (matrix) of size (p+1) x (m+1)
    # np.zeros is used for efficient initialization.
    dp = np.zeros((p + 1, m + 1), dtype=int)

    # Initialize the first column. This represents the cost of deleting
    # characters from the 'term' to match an empty string.
    for i in range(p + 1):
        dp[i, 0] = i

    # The first row is already initialized to zeros by np.zeros.
    # This is the key modification for substring search: an empty term
    # can match any prefix of the sentence with zero cost, allowing a
    # match to start anywhere.

    # --- 2. Fill the DP Table ---
    for i in range(1, p + 1):
        for j in range(1, m + 1):
            # If characters are the same, the cost is 0, otherwise it's 1.
            cost = 0 if term[i - 1] == sentence[j - 1] else 1

            # Calculate the cost for each operation
            insertion_cost = dp[i, j - 1] + 1
            deletion_cost = dp[i - 1, j] + 1
            substitution_cost = dp[i - 1, j - 1] + cost

            # The value of dp[i, j] is the minimum of these three costs
            dp[i, j] = min(insertion_cost, deletion_cost, substitution_cost)

    # --- 3. Find the Result in the Last Row ---
    # The last row of the DP table (dp[p]) contains the edit distances for
    # matching the entire 'term' against all substrings of the 'sentence'.
    min_diff = np.min(dp[p, :])
    
    # Find the position of the first occurrence of the minimum difference
    # We add 1 to j because argmin returns a 0-based index
    end_pos = -1
    if min_diff <= n:
        # np.argmin gives the first index of the minimum value
        end_pos = np.argmin(dp[p, :])


    # --- 4. Return the Final Result ---
    # Check if the minimum difference found is within the allowed threshold 'n'.
    is_match_found = min_diff <= n

    return is_match_found

def calculate_skewed_penalty(x, center=70, skewness=5, scale=200):
    """
    Calculate penalty value based on skewed normal distribution.

    Args:
        x: Input value(s) to evaluate
        center: Location parameter (center/mode of distribution)
        skewness: Controls skew direction and magnitude (negative = left skew)
        scale: Controls width of distribution

    Returns:
        y: Penalty value(s) corresponding to input x
    """
    max_y = 0.0036056853513317067  # base on center=70, skewness=5, scale=200
    return skewnorm.pdf(x, a=skewness, loc=center, scale=scale)/max_y


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

def llm_check(query, prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    score = 0.
    data = {"query": query, "gold_answer":"", "answer":prediction}
    for golden_answer in golden_answers:
        data["gold_answer"] = golden_answer
        res = process_row(data)
        if res["grade_description"] == "CORRECT":
            score = 1.0
            return score
    return score

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0.
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1.0
            break
    return score


def find_num_tags(text: str, tag: str) -> int:
    """
    Finds all content between <smt>...</smt> tag pairs in a text string.

    Args:
        text: Input string to search for tags

    Returns:
        List of strings containing all text between tag pairs
    """
    pattern = rf'<{tag}>(.*?)</{tag}>'
    # pattern = rf'<{tag}>\s*(\{{.*?\}})\s*</{tag}>'
    if tag == "tool_call":
        pattern = rf'<{tag}>\s*(\{{\s*".*?"\s*:\s*.+?\s*}})\s*</{tag}>'
    return len(re.findall(pattern, text, flags=re.DOTALL))


class ToolRubric(Rubric):

    def __init__(
        self,
        parser: Parser = XMLParser(fields=["think", ("tool_call","answer")]),# 
        env_parser: Parser = XMLParser(fields=["tool_response"]),
        tools: List[Callable] = [],
    ):
        super().__init__(parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {
            tool.__name__ if hasattr(tool, "__name__") else str(tool): tool
            for tool in tools
        }
        self.reward_funcs = [
            self.correct_answer_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.efficient_thinking_reward_func,
            self.num_xml_reward_func,
            self.visit_tool_reward_func,
            self.avg_thinking_length_func
        ]
        self.reward_weights = [
            1.0,
            0.2,
            0.2,
            0.1,
            0.1,
            0.6,
            0.
        ]
        self.think_mode = os.environ.get("THINK_MODE", True)
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

        except Exception:
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

    def correct_answer_reward_func(self,prompt, completion, answer, task, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        if self.think_mode:
            check_xml = self.num_xml_reward_func(completion=completion)
        else:
            check_xml = 1.
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
            reward = self.qa_reward_func(prompt,
                completion, answer, task, **kwargs)  # * check_xml
        else:
            reward = 0.0
        return reward

    def avg_thinking_length_func(self, completion,  **kwargs) -> float:
        num_think_tag = 0.
        total_token = 0.
        for compl in completion:
            if compl.get("role") == "assistant":
                content = compl.get("content", "")
                # If the message doesn't call tool or contains answer then we don't calculate reward for it
                if find_num_tags(content, "tool_call") == 0 or find_num_tags(content, "answer") > 0:
                    continue
                # invalid format also reward 0.
                if content.count("<think>") != content.count("</think>") and (content.count("<think>") > 0 or content.count("</think>") > 0):
                    return 0.

                if "[ERROR] max_tokens_reached" in content:
                    return 0.

                think_tags = re.findall(
                    r'<think>(.*?)</think>', content, re.DOTALL)
                for content_ in think_tags:
                    # Remove leading/trailing whitespace and split into tokens
                    tokens = content_.strip().split()
                    num_think_tag += 1
                    total_token += len(tokens)
        if num_think_tag > 0:
            avg_length = total_token/num_think_tag
            return avg_length
        return 0.

    def efficient_thinking_reward_func(self, completion,  **kwargs) -> float:
        num_think_tag = 0.
        total_token = 0.
        for compl in completion:
            if compl.get("role") == "assistant":
                content = compl.get("content", "")
                # If the message doesn't call tool or contains answer then we don't calculate reward for it
                if find_num_tags(content, "tool_call") == 0 or find_num_tags(content, "answer") > 0:
                    continue
                # invalid format also reward 0.
                if content.count("<think>") != content.count("</think>") and (content.count("<think>") > 0 or content.count("</think>") > 0):
                    return 0.

                if "[ERROR] max_tokens_reached" in content:
                    return 0.

                think_tags = re.findall(
                    r'<think>(.*?)</think>', content, re.DOTALL)
                for content_ in think_tags:
                    # Remove leading/trailing whitespace and split into tokens
                    tokens = content_.strip().split()
                    num_think_tag += 1
                    total_token += len(tokens)
        if num_think_tag > 0:
            avg_length = total_token/num_think_tag
            return calculate_skewed_penalty(avg_length)
        return 0.

    def num_xml_reward_func(self, completion, **kwargs) -> float:
        import re
        num_tool = 0.
        num_think = 0.
        num_answer = 0.
        total_reward = 0.
        num_turn = 0.
        for compl in completion:
            if compl.get("role") == "assistant":
                num_turn += 1
                content = compl.get("content", "")
                num_tool += find_num_tags(content, "tool_call")
                num_think += find_num_tags(content, "think")
                num_answer += find_num_tags(content, "answer")
                if "[ERROR] max_tokens_reached" in content:
                    return 0.
                if content.count("<tool_call>") != content.count("</tool_call>") and (content.count("<tool_call>") > 0 or content.count("</tool_call>") > 0):
                    return 0.
                if content.count("<think>") != content.count("</think>") and (content.count("<think>") > 0 or content.count("</think>") > 0):
                    return 0.
                if content.count("<answer>") != content.count("</answer>") and (content.count("<answer>") > 0 or content.count("</answer>") > 0):
                    # total_reward -= 2*abs(content.count("<answer>") - content.count("</answer>") )
                    return 0.
                if (find_num_tags(content, "tool_call") + find_num_tags(content, "answer") == 0):
                    return 0.
                if find_num_tags(content, "tool_call") > 0 and find_num_tags(content, "answer") > 0:
                    return 0.  # total_reward -= 2
                if find_num_tags(content, "tool_call") > 0 and find_num_tags(content, "tool_response") > 0:
                    return 0.  # total_reward -= 2
        if num_turn > 0:
            total_reward += num_answer*(num_think + num_tool)/num_turn # 
        return total_reward

    def qa_reward_func(self,prompt, completion, answer, task, **kwargs) -> float | None:
        """
        Reward function that checks if the QA answer matches the expected answer.
        Uses text normalization and either exact matching or substring matching.
        """
        match_mode = "llm"
        query = ""
        for obj in prompt:
            if obj["role"] == "user":
                query = obj["content"]
        if query == "":
            raise ("error")
        if task == "qa":
            response =  str(self.parser.parse_answer(completion)) #completion[-1]["content"] #

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
                reward = max(subem_check(response, ans)
                                    for ans in answers) 
            elif match_mode == "llm":
                reward = max(llm_check(query, response, ans)
                                    for ans in answers)   
                
            
            else:
                reward = 1.0 if any(em_check(response, ans)
                                    for ans in answers) else 0.0
        else:
            reward = None

        return reward

    def visit_tool_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        tool_attempts = 0
        successful_executions = 0
        set_tools = set()
        num_visit = 0
        num_search = 0

        # Find assistant messages with tools and their responses
        for i, msg in enumerate(completion):
            if msg['role'] == 'assistant':
                # Use parser to check for tool tag
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'tool_call') and parsed.tool_call is not None:
                    # Found a properly formatted tool message
                    if i + 1 < len(completion) and completion[i + 1]['role'] == 'user':
                        tool_attempts += 1
                        # Check response with env_parser
                        parsed_response = self.env_parser.parse(
                            completion[i + 1]['content'])
                        if hasattr(parsed_response, 'tool_response') and parsed_response.tool_response is not None and not parsed_response.tool_response.startswith("Error:"):
                            successful_executions += 1
                            tool_name = json.loads(
                                parsed.tool_call).get("name")
                            if tool_name == "visit_tool":
                                num_visit += 1
                            elif tool_name == "web_search":
                                num_search += 1

        # Calculate reward
        if tool_attempts == 0 or num_search == 0:
            return -0.5
        # return num_visit/num_search
        x = num_visit/num_search
        if x < 1:
            return -0.5
        else:
            return ((x-1)/4)**0.25
        
    def tool_execution_reward_func(self, completion: List[Dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        tool_attempts = 0
        successful_executions = 0
        set_tools = set()

        # Find assistant messages with tools and their responses
        for i, msg in enumerate(completion):
            if msg["role"] == "assistant":
                # Use parser to check for tool tag
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'tool_call') and parsed.tool_call is not None:

                    # Found a properly formatted tool message
                    if i + 1 < len(completion) and completion[i + 1]["role"] == "user":
                        tool_attempts += 1
                        # Check response with env_parser
                        parsed_response = self.env_parser.parse(
                            completion[i + 1]["content"]
                        )
                        if (
                            hasattr(parsed_response, "tool_response")
                            and parsed_response.tool_response is not None
                            and not parsed_response.tool_response.startswith("Error:")
                        ):
                            successful_executions += 1
                            tool_name = json.loads(
                                parsed.tool_call).get("name")
                            if tool_name:
                                set_tools.add(tool_name)

        # Calculate reward
        if tool_attempts == 0:
            return 0.0
        if self.think_mode:
            check_xml = self.num_xml_reward_func(completion=completion)
        else:
            check_xml = 1.
        # * check_xml
        return (successful_executions * len(set_tools) / tool_attempts)

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
                if msg["role"] == "assistant":
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool_call') and parsed.tool_call is not None:
                        try:
                            command = json.loads(parsed.tool_call)
                            if isinstance(command, dict) and command.get("name") == tool_name:

                                # Found a properly formatted tool message for the specific tool
                                if (
                                    i + 1 < len(completion)
                                    and completion[i + 1]["role"] == "user"
                                ):
                                    tool_attempts += 1
                                    # Check response with env_parser
                                    parsed_response = self.env_parser.parse(
                                        completion[i + 1]["content"]
                                    )
                                    if (
                                        hasattr(parsed_response, "tool_response")
                                        and parsed_response.tool_response is not None
                                        and not parsed_response.tool_response.startswith(
                                            "Error:"
                                        )
                                    ):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass

            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return successful_executions / tool_attempts

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
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool_call") and parsed.tool_call is not None:
                        try:
                            command = json.loads(parsed.tool_call)
                            if (
                                isinstance(command, dict)
                                and command.get("name") == tool_name
                            ):
                                # Found a properly formatted tool message for the specific tool
                                if (
                                    i + 1 < len(completion)
                                    and completion[i + 1]["role"] == "user"
                                ):
                                    parsed_response = self.env_parser.parse(
                                        completion[i + 1]["content"]
                                    )
                                    if (
                                        hasattr(parsed_response, "tool_response")
                                        and parsed_response.tool_response is not None
                                        and not parsed_response.tool_response.startswith(
                                            "Error:"
                                        )
                                    ):
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

        def tool_attempt_reward_func(
            completion: List[Dict[str, str]], **kwargs
        ) -> float:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            attempted_executions = 0.0
            for i, msg in enumerate(completion):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool_call") and parsed.tool_call is not None:
                        try:
                            command = json.loads(parsed.tool_call)
                            if (
                                isinstance(command, dict)
                                and command.get("name") == tool_name
                            ):
                                attempted_executions += 1
                        except json.JSONDecodeError:
                            pass
            return attempted_executions

        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func
