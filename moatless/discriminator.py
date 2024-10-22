import logging
from typing import Optional, List, Callable, Tuple
from functools import partial

from instructor import OpenAISchema
from pydantic import Field, BaseModel

from moatless.node import Node
from moatless.completion.model import Completion, Message, UserMessage
from moatless.debate import MultiAgentDebate

logger = logging.getLogger(__name__)


class Discriminator(BaseModel):
    def select(self, nodes: List[Node]) -> Node | None:
        raise NotImplementedError

class MeanAwardDiscriminator(Discriminator):
    def select(self, nodes: List[Node]) -> Node | None:
        best_finish_node: Node | None = None
        best_mean_reward = float("-inf")
        trajectories_mean_rewards = []

        for finished_node in nodes:
            mean_reward = finished_node.calculate_mean_reward()

            trajectories_mean_rewards.append((finished_node.node_id, mean_reward))
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_finish_node = finished_node

        logger.info(
            f"Mean Rewards for finished trajectories: {trajectories_mean_rewards}"
        )

        if best_finish_node:
            logger.info(
                f"Best finished path finished on Node{best_finish_node.node_id} with mean reward: {best_mean_reward}"
            )
            return best_finish_node
        else:
            logger.info(
                "No valid finished path found. This should not happen if there are finished nodes."
            )
            return None


class Discriminator:
    def select(self, nodes: List[Node]) -> Node | None:
        raise NotImplementedError

class AgentDiscriminatorChoice(OpenAISchema):
    ID: int
    EXPLANATION: str

class AgentDiscriminator(Discriminator):
    def __init__(self, 
                 create_completion: Callable[[List[Message], str, List[type[OpenAISchema]] | None], Tuple[OpenAISchema, Completion]],
                 debate_settings):
        self.create_completion = create_completion
        self.debate = MultiAgentDebate(
            n_agents=debate_settings.n_agents,
            n_rounds=debate_settings.n_rounds,
        )

    def select(self, nodes: List[Node]) -> Node | None:
        best_finish_node: Node | None = None
        best_resolved_status = False
        nodes_results = []

        for finished_node in nodes:
            comparison_result = self.compare_solutions_v2(finished_node)

            if comparison_result:
                resolved_status = comparison_result.observation.extra.get("evaluation_result", {}).get("resolved", False)
                status_details = f"Status: {comparison_result.observation.extra.get('evaluation_result', {}).get('tests_status', {}).get('status', 'Unknown')}"
            else:
                resolved_status = False
                status_details = "No valid comparison result found"

            nodes_results.append((finished_node.node_id, resolved_status, status_details))

            if resolved_status and (not best_resolved_status or finished_node.calculate_mean_reward() > best_finish_node.calculate_mean_reward()):
                best_resolved_status = True
                best_finish_node = finished_node

        logger.info(f"Discriminator results for finished nodes: {nodes_results}")

        if best_finish_node:
            logger.info(
                f"Best finished path finished on Node{best_finish_node.node_id} with resolved status: {best_resolved_status}"
            )
            return best_finish_node
        else:
            logger.info(
                "No valid finished path found. This should not happen if there are finished nodes."
            )
            return None

    def compare_solutions_v1(self, solutions, problem_statement, debate=False):
        ROLE_PROMPT = f"""Below are a series of suggested changes to address the <Problem Statement>.   
Your task is to carefully evaluate each change and decide which one is the most appropriate to address the issue."""
        FORMAT_PROMPT = f"""Provide your answer in the following format:
<Explanation>: A comprehensive explanation and reasoning behind your decision
<ID>: The ID of the change you believe is the most appropriate"""

        SYSTEM_MESSAGE = f"{ROLE_PROMPT}\n{FORMAT_PROMPT}"
        USER_MESSAGE = f"<Problem Statement>\n{problem_statement}</Problem Statement>\n<Solutions>\n{solutions}\n</Solutions>"

        messages = [
            Message(role="system", content=SYSTEM_MESSAGE),
            UserMessage(content=USER_MESSAGE)
        ]

        if debate:
            completion_func = partial(
                self.model_completion.get_completion,
                messages=messages,
                system_prompt=SYSTEM_MESSAGE,
            )
            response, completion, messages = self.debate.conduct_debate(messages=messages,
                                                                model=self.model,
                                                                completion_func=completion_func,
                                                                output_format=AgentDiscriminatorChoice)
        else:
            response, completion  = self.model_completion.get_completion(messages)

        return response.ID, response.EXPLANATION
    
    def compare_solutions_v2(self, node: Node, include_history: bool = False, 
                             show_reward: bool = True, debate: bool = False) -> Node | None:
        finished_nodes = [
            n for n in node.get_trajectory()
            if n.action.name == "Finish" and
               n.file_context and
               n.file_context.generate_git_patch()
        ]

        if len(finished_nodes) == 0:
            logger.warning(f"No finished solutions found")
            return None
        elif len(finished_nodes) == 1:
            return finished_nodes[0]
        else:
            solutions = self.create_message_compare_solutions(finished_nodes, include_history, show_reward)
            node_id, explanation = self.compare_solutions_v1(solutions, node.get_root().message, debate=debate)

        if not node_id:
            logger.warning(f"Failed to find a valid node_id, return best_node")
            return None

        return next((n for n in finished_nodes if n.node_id == node_id), None)

    def create_message_compare_solutions(self, finished_nodes, include_history: bool = False, show_reward: bool = False):
        logger.info(f"Comparing {len(finished_nodes)} solutions.")

        solutions = ""
        for finished_node in finished_nodes:
            solutions += f"\n<Solution id={finished_node.node_id}>\n"

            if show_reward:
                reward = finished_node.reward
                if reward:
                    solutions += f"<Explanation>{reward.explanation}</Explanation>\n"
                    solutions += f"<Reward>{reward.value}</Reward>\n"

            if include_history: 
                node_history = finished_node.get_trajectory()[:-1]  # Exclude the current node
                if node_history:
                    formatted_history = []
                    counter = 0

                    for previous_node in node_history:
                        if previous_node.action.name in ["Analyze", "Implement", "Test"]:  # Replace with actual action names to explore
                            counter += 1
                            formatted_state = f"\n# {counter}. Action: {previous_node.action.name}\n\n"
                            formatted_state += previous_node.action.to_prompt()
                            formatted_history.append(formatted_state)

                    if formatted_history:
                        solutions += "<history>\n"
                        solutions += "\n".join(formatted_history)
                        solutions += "\n</history>\n\n"

            solutions += "<Patch>"
            solutions += finished_node.file_context.generate_git_patch()
            solutions += "</Patch>"

            solutions += "\n</Solution>\n"
        return solutions
