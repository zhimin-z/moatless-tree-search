import logging
import os

from moatless.benchmark.evaluation import Evaluation
from moatless.benchmark.run_evaluation import evaluate_search_and_code, django_dataset, one_easy_per_repo
from moatless.edit.edit import EditCode
from moatless.edit.plan_v2 import PlanToCode
from moatless.find import IdentifyCode, SearchCode
from moatless.transition_rules import TreeSearchSettings, AgenticLoopSettings
from moatless.transitions import search_and_code_transitions_v2

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

repo_base_dir = "/tmp/repos"
evaluations_dir = os.path.join(os.getenv("MOATLESS_DIR", "/home/albert/repos/albert/sw-planner-2/trajs"), "evaluations")

tree_search_settings = TreeSearchSettings(
    max_expansions=3,
    max_iterations=50,
    min_finished_transitions=3,
    max_finished_transitions=5,
    reward_threshold=90,
    provide_feedback=True,
    debate=False,
    value_function_model="gpt-4o-mini-2024-07-18",
    value_function_model_temperature=0.0
)


def test_gpt4o_mini():
    evaluate_search_and_code(
        model="gpt-4o-mini-2024-07-18",
        temperature=1.0,
        instance_ids=["django__django-11039"], # django_dataset,
        max_transitions=30,
        enable_mcts=True,
        tree_search_settings=tree_search_settings,
        name="django",
        evaluation_name="20240918_django_search_feedback",
        num_workers=1
    )


def test_gpt4o_mini_easy_per_repo():
    global_params = {
        "model": "deepseek/deepseek-chat",
        "temperature": 1.6,
        "max_tokens": 4000,
        "max_prompt_file_tokens": 16000,
    }

    state_params = {
        SearchCode: {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 1.0
        },
        IdentifyCode: {
            "model": "gpt-4o-mini-2024-07-18",
            "temperature": 0.0
        },
        PlanToCode: {
            "model": "deepseek/deepseek-chat",
            "temperature": 1.6,
            "tool_model": "gpt-4o-mini-2024-07-18",
            "use_completion_message_history": True,
        },
        EditCode: {
            "model": "deepseek/deepseek-chat",
            "temperature": 0.0
        }
    }

    loop_settings = AgenticLoopSettings(
        max_cost=0.5,
        max_transitions=100
    )

    evaluation = Evaluation(
        transitions=search_and_code_transitions_v2(
            tree_search_settings=tree_search_settings,
            loop_settings=loop_settings,
            global_params=global_params,
            state_params=state_params
        ),
        evaluations_dir=evaluations_dir,
        evaluation_name="deepseek",
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        enable_mcts=True,
        num_workers=1,
        detailed_report=True,
        use_testbed=True
    )

    evaluation.run_evaluation(
        # resolved_by=resolved_by,
        instance_ids=["django__django-12453"],
    )


