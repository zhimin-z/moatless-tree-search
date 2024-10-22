import logging
from typing import Optional, List

from instructor import OpenAISchema
from pydantic import Field

from moatless.node import Node
from moatless.search.reward import LLM_Value_Function
from moatless.trajectory import Trajectory, TrajectoryState
from moatless.completion import ModelCompletion, LLMResponseFormat
from moatless.schema import Message, OpenAISchema


logger = logging.getLogger(__name__)


class Discriminator:
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

class DiscriminatorChoice(OpenAISchema):
    id: int
    explanation: str

class DiscriminatorAgent(Discriminator):
    def __init__(self, create_completion: Callable[[List[Message], str, List[type[OpenAISchema]] | None], Tuple[OpenAISchema, Completion]]):
        self.create_completion = create_completion

    def select(self, nodes: List[Node]) -> Node | None:
        best_finish_node: Node | None = None
        best_resolved_status = False
        trajectories_results = []

        for finished_node in nodes:
            trajectory = finished_node.trajectory
            transition = self.compare_solutions2(trajectory)

            if transition:
                resolved_status = transition.state.output.get("evaluation_result", {}).get("resolved", False)
                status_details = f"Status: {transition.state.output.get('evaluation_result', {}).get('tests_status', {}).get('status', 'Unknown')}"
            else:
                resolved_status = False
                status_details = "No valid transition found"

            trajectories_results.append((finished_node.node_id, resolved_status, status_details))

            if resolved_status and (not best_resolved_status or finished_node.calculate_mean_reward() > best_finish_node.calculate_mean_reward()):
                best_resolved_status = True
                best_finish_node = finished_node

        logger.info(f"Discriminator results for finished trajectories: {trajectories_results}")

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
            Message(role="user", content=USER_MESSAGE)
        ]

        action_request, completion_response = self.create_completion(
            messages=messages,
            system_prompt=SYSTEM_MESSAGE,
            actions=[DiscriminatorChoice]
        )

        return action_request.id, action_request.explanation
    
    def compare_solutions2(self, trajectory: Trajectory, include_history: bool = False, 
                           show_reward: bool = True, debate: bool = False) -> TrajectoryState | None:
        finished_solutions = [
            transition
            for transition in trajectory.transitions
            if transition.state.name == "Finished" and
               transition.snapshot["repository"] and
               transition.snapshot["repository"].get("patch")
        ]

        if len(finished_solutions) == 0:
            logger.warning(f"No finished solutions found")
            return None
        elif len(finished_solutions) == 1:
            return finished_solutions[0]
        else:
            solutions = self.create_message_compare_solutions(finished_solutions, include_history, show_reward)
            state_id, explanation = self.compare_solutions_v1(solutions, trajectory.initial_message, debate=debate)

        if not state_id:
            logger.warning(f"Failed to find a valid state_id, return best_trajectory")
            return None

        return next((transition for transition in finished_solutions if transition.state.id == state_id), None)

    def create_message_compare_solutions(self, finished_solutions, include_history: bool = False, show_reward: bool = False):
        logger.info(f"Comparing {len(finished_solutions)} solutions.")

        solutions = ""
        for finished_solution in finished_solutions:
            solutions += f"\n<Solution id={finished_solution.state.id}>\n"

            if show_reward:
                visit = next((visit for visit in finished_solution.state.visits if visit.source_state_id == finished_solution.state.id), None)
                if visit:
                    solutions += f"<Explanation>{visit.explanation}</Explanation>\n"
                    solutions += f"<Reward>{visit.value}</Reward>\n"

            if include_history: 
                state_history = finished_solution.state.get_previous_states()
                if state_history:
                    formatted_history = []
                    counter = 0

                    for previous_state in state_history:
                        if previous_state.name in ["Analyze", "Implement", "Test"]:  # Replace with actual states to explore
                            counter += 1
                            formatted_state = f"\n# {counter}. Action: {previous_state.action_request.name}\n\n"
                            formatted_state += previous_state.action_request.to_prompt()
                            formatted_history.append(formatted_state)

                    if formatted_history:
                        solutions += "<history>\n"
                        solutions += "\n".join(formatted_history)
                        solutions += "\n</history>\n\n"

            solutions += "<Patch>"
            solutions += finished_solution.snapshot["repository"].get("patch")
            solutions += "</Patch>"

            solutions += "\n</Solution>\n"
        return solutions