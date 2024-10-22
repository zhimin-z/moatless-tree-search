import collections
from typing import List, Callable, Tuple
import json
from tqdm import tqdm

from instructor import OpenAISchema
from litellm import token_counter

from utils.misc import save_to_json
from completion import CompletionModel, Completion, UserMessage, Message

from pydantic import Field
from moatless.settings import LLMResponseFormat

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

                assistant_message = self.format_assistant_message(action_request, completion)
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
            final_action, final_completion, final_messages = self.generate_conclusion(
                messages[-1].content,
                debate_summary,
                messages,
                create_completion,
                output_format
            )
        else:
            final_action, final_completion, final_messages = None, None, messages

        debate_log["conclusion"] = final_completion.choices[0].message
        
        # Calculate token usage
        prompt_tokens = token_counter(text=str(debate_log["messages"]))
        completion_tokens = token_counter(text=str(final_messages) if final_messages else " ")
        total_tokens = prompt_tokens + completion_tokens

        if not node_id:
            node_id = str(len(self.debates) + 1)
        self.debates[node_id].append(debate_log)

        if hasattr(self, "debate_log_dir"):
            logger.info(f"Saving debate log to {self.debate_log_dir}")
            save_to_json(self.debates, self.debate_log_dir)

        return final_action, final_completion, final_messages

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

        return action_request, completion, completion.choices[0].message.content

    def format_assistant_message(self, action_request, completion):
        if action_request:
            return Message(role="assistant", content=json.dumps(action_request.dict()))
        elif completion.choices and completion.choices[0].message:
            return Message(role="assistant", content=completion.choices[0].message.content)
        else:
            return None

    def construct_debate_summary(self, agent_contexts):
        summary = "Debate Summary:\n\n"
        for i, context in enumerate(agent_contexts):
            summary += f"Agent {i+1} final response:\n{context[-1].content}\n\n"
        return summary


if __name__ == "__main__":
    model = "openai/Qwen/Qwen2.5-72B-Instruct"
    system_message = "You are a highly capable AI assistant tasked with synthesizing information and reaching conclusions."
    prompt = "What should we name you guys? Be creative and funny, and mysterious."

    class NameSuggestion(ValueFunctionDebateConclusion):
        name: str = Field(description="The name you think is the best.")

    # Create ModelCompletion instance
    model_completion = CompletionModel(
        model=model,
        temperature=1.0,
        max_tokens=1000,
        response_format=LLMResponseFormat.JSON
    )

    # Prepare messages
    messages = [
        Message(role="system", content=system_message),
        UserMessage(content=prompt)
    ]

    # Create MultiAgentDebate instance
    debate = MultiAgentDebate(n_agents=8, n_rounds=3)

    # Conduct the debate
    final_action, final_completion, final_messages = debate.conduct_debate(
        messages=messages,
        create_completion=model_completion.create_completion,
        output_format=NameSuggestion,
    )

    # Generate conclusion
    conclusion_prompt = f"Based on the debate, summarize all the names and choose the best one according to the arguments made. Debate summary: {final_messages}"
    conclusion_messages = [
        Message(role="system", content=system_message),
        UserMessage(content=conclusion_prompt)
    ]

    conclusion_action, conclusion_completion = model_completion.create_completion(
        messages=conclusion_messages,
        system_prompt=system_message,
    )

    # Print results
    print("Debate summary:")
    for message in final_messages:
        if isinstance(message, UserMessage):
            print(f"User: {message.content}")
        else:
            print(f"{message.role.capitalize()}: {message.content}")

    print("\nConclusion:")
    print(f"Best name: {conclusion_action.best_name}")
    print(f"Summary: {conclusion_action.summary}")