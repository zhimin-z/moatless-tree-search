import os
import collections
from typing import List, Callable, Tuple
import json
from tqdm import tqdm

from instructor import OpenAISchema
from litellm import token_counter

from moatless.utils.misc import save_to_json
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import Completion, Message, UserMessage

from pydantic import Field

import logging
logger = logging.getLogger(__name__)

CONCLUSION_PROMPT = """Based on the initial problem context and the answers from the debate of the other agents, construct an optimal answer.
Consider the different perspectives presented by the agents and the context provided in order to reach the correct conclusion.
Do not refer to the participants, but rather just report your recommendations as if they were your own.
Strictly adhere to any output format used in the Agent responses, and especially any tool/api/function calls if present, like those enclosed for example those enclosed in angle brackets i.e <tool_call> or **value**.
"""

VALUE_OUTPUT_FORMAT = """OUTPUT FORMAT:

<Explanation>: 2-3 sentences explaining the the reasoning in your decision, alluding to the *common mistakes* where appropriate.
<Reward>: integer reward (range: -100 to 100)."""

class ValueFunctionDebateConclusion(OpenAISchema):
    explanation: str = Field(
        description="2-3 sentences explaining the the reasoning in your decision, alluding to the *common mistakes* where appropriate."
    )
    reward: int = Field(
        description="integer reward (range: -100 to 100)."
    )

class MultiAgentDebate:
    def __init__(self, n_agents=8, n_rounds=3, **kwargs):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.include_conclusion = True
        self.debates = collections.defaultdict(list)
        self.temperature = 1
        self.model_agents_map = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.completion_params = {'temperature': self.temperature}

    def conduct_debate(self, 
                       messages: List[Message], 
                       create_completion: Callable[[List[Message], str, List[type[OpenAISchema]] | None], Tuple[OpenAISchema, Completion]],
                       output_format: type[OpenAISchema], 
                       model: str = None, 
                       **kwargs):
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        node_id = kwargs.get("node_id", None)
        system_message = next((msg.content for msg in messages if msg.role == 'system'), None)
        if not system_message:
            raise ValueError("No system message found in the conversation history.")

        # Initialize agent contexts with the full conversation history
        agent_contexts = [messages.copy() for _ in range(self.n_agents)]

        debate_log = {
            "messages": messages,
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
            "rounds": []
        }

        for round in tqdm(range(self.n_rounds), desc="Debate Rounds"):
            round_log = {"round": round, "agent_responses": []}

            for agent, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:agent] + agent_contexts[agent+1:]
                    debate_message = self.construct_debate_message(agent_contexts_other, -1)
                    agent_context.append(debate_message)

                action_request, completion = create_completion(
                    messages=agent_context,
                    system_prompt=system_message,
                    actions=[output_format]
                )

                assistant_message = self.format_agent_message(action_request, completion)
                agent_context.append(assistant_message)

                round_log["agent_responses"].append({
                    "agent": agent,
                    "message": debate_message if round != 0 else None,
                    "response": assistant_message,
                })

            debate_log["rounds"].append(round_log)

        debate_summary = self.construct_debate_summary(agent_contexts)
        debate_log["summary"] = debate_summary

        if self.include_conclusion:
            final_action, final_completion, conclusion_dict = self.generate_conclusion(
                messages[-1].content,
                debate_summary,
                messages,
                create_completion,
                output_format
            )
        else:
            final_action, final_completion, conclusion_dict = None, None, messages

        if conclusion_dict:
            debate_log["conclusion"] = conclusion_dict
        elif isinstance(final_completion, dict) and 'choices' in final_completion:
            debate_log["conclusion"] = final_completion['choices'][0]['message']
        else:
            debate_log["conclusion"] = "No conclusion available"        
            
        # Calculate token usage
        prompt_tokens = token_counter(text=str(debate_log["messages"]))
        completion_tokens = token_counter(text=str(conclusion_dict) if conclusion_dict else " ")
        total_tokens = prompt_tokens + completion_tokens

        if not node_id:
            node_id = str(len(self.debates) + 1)
        self.debates[node_id].append(debate_log)

        if hasattr(self, "debate_log_dir"):
            logger.info(f"Saving debate log to {self.debate_log_dir}")
            save_to_json(self.debates, self.debate_log_dir)

        return final_action, final_completion, conclusion_dict

    def construct_debate_message(self, agents, idx):
        prefix_string = "These are the solutions to the problem from other agents: "

        for agent_num, agent in enumerate(agents):
            if idx < len(agent):
                agent_response = agent[idx].content
                response = f"\n\nAgent {agent_num + 1} solution: ```{agent_response}```"
                prefix_string += response
            else:
                print(f"Warning: Agent {agent_num} does not have a response at index {idx}")

        prefix_string += """\n\nGiven the provided context and responses, provide your own response.
                                You can first reason about the solutions provided by the other agents and then provide your own solution. 
                                Strictly adhere to any output format used in the responses, and especially any tool/api/function calls if present, like those enclosed in <> or **."""
        return Message(role="user", content=prefix_string)

    def generate_conclusion(self, 
                            initial_context, 
                            debate_summary, 
                            messages, 
                            create_completion, 
                            output_format):
        conclusion_prompt = f"""
        **Initial problem context:**
        {initial_context}

        **Agent Answers:**
        {debate_summary}

        {output_format.__doc__}
        """

        conclusion_context = [
            Message(role="system", content="You are a highly capable AI assistant tasked with synthesizing information and reaching conclusions."),
            Message(role="user", content=conclusion_prompt)
        ]

        action_request, completion = create_completion(
            messages=conclusion_context,
            system_prompt="You are a highly capable AI assistant tasked with synthesizing information and reaching conclusions.",
            actions=[output_format]
        )

        if isinstance(completion, dict) and 'choices' in completion:
            choice = completion['choices'][0]
            if 'message' in choice and 'tool_calls' in choice['message']:
                tool_call = choice['message']['tool_calls'][0]
                if 'function' in tool_call and 'parsed_arguments' in tool_call['function']:
                    parsed_args = tool_call['function']['parsed_arguments']
                    conclusion = output_format(**parsed_args)
                    return action_request, completion, conclusion

        # Fallback if the expected structure is not found
        return action_request, completion, None

    def format_agent_message(self, action_request, completion):
        if action_request:
            content = json.dumps({
                "action_request": action_request.__class__.__name__,
                "arguments": action_request.dict()
            })
            return Message(role="user", content=content)
        elif completion.choices and completion.choices[0].message:
            return Message(role="user", content=completion.choices[0].message.content)
        else:
            return None

    def construct_debate_summary(self, agent_contexts):
        summary = "Debate Summary:\n\n"
        for i, context in enumerate(agent_contexts):
            summary += f"Agent {i+1} final response:\n{context[-1].content}\n\n"
        return summary


if __name__ == "__main__":
    # setup your desired output format
    class NameConclusion(OpenAISchema):
        explanation: str = Field(description="2-3 sentences explaining the the reasoning in your decision.")
        conclusion: str = Field(description="The name you think is the best")

    model = "openai/Qwen/Qwen2.5-72B-Instruct"
    # model = "gpt-4o-mini"
    system_message = "You are a highly capable AI assistant tasked with synthesizing information and reaching conclusions."
    prompt = "What should we name you guys? Be creative and funny, and mysterious."


    # Prepare common parameters for CompletionModel
    common_params = {
        'model': model,
        'temperature': 1.0,
        'max_tokens': 1000,
        'response_format': LLMResponseFormat.JSON
    }

    # Add CUSTOM_LLM_API_KEY if it exists and the model starts with "openai"
    if os.getenv('CUSTOM_LLM_API_KEY') and model.startswith("openai"):
        print(f"Using custom API key for model: {model}")
        common_params['model_api_key'] = os.getenv('CUSTOM_LLM_API_KEY', None)
        common_params['model_base_url'] = os.getenv('CUSTOM_LLM_API_BASE', None)

    # Create ModelCompletion instance with the updated parameters
    model_completion = CompletionModel(**common_params)

    # Prepare messages
    messages = [
        Message(role="system", content=system_message),
        UserMessage(content=prompt)
    ]

    # Create MultiAgentDebate instance
    debate = MultiAgentDebate(n_agents=8, n_rounds=2)

    # Conduct the debate
    final_action, final_completion, final_messages = debate.conduct_debate(
        messages=messages,
        create_completion=model_completion.create_completion,
        output_format=NameConclusion,
    )

    print(final_action.explanation)
    print(final_action.conclusion)