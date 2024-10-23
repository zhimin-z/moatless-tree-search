import argparse
import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

from moatless.benchmark.evaluation import create_evaluation_name, Evaluation, TreeSearchSettings, ModelSettings

logger = logging.getLogger(__name__)


def evaluate_search_and_code(
    resolved_by: Optional[int] = None,
    instance_ids: Optional[list] = None,
    repo_base_dir: str | None = None,
    use_testbed: bool = True,
    num_workers: int = 4,
    evaluation_name=None,
    evaluations_dir=None,
    date=None,
    tree_search_settings: TreeSearchSettings = None,
    **kwargs,
):
    temperature = tree_search_settings.model.temperature or kwargs.get("temp_bias", 0.2)

    if evaluation_name is None:
        evaluation_name = create_evaluation_name(
            model=tree_search_settings.model.model,
            date=date,
            max_expansions=tree_search_settings.max_expansions,
            debate=tree_search_settings.debate,
            provide_feedback=tree_search_settings.provide_feedback,
            temp_bias=temperature,
            use_testbed=use_testbed,
        )

    if not evaluations_dir:
        evaluations_dir = os.getenv("MOATLESS_DIR")

    # Expect models with prefix openai/ to be custom
    if tree_search_settings.model.model.startswith("openai/"):
        tree_search_settings.model.base_url = os.getenv("CUSTOM_LLM_API_BASE")
        tree_search_settings.model.api_key = os.getenv("CUSTOM_LLM_API_KEY")

    logger.info("Evaluation Parameters:")
    logger.info(f"Evalation dir: {evaluations_dir}")
    logger.info(f"Evaluation Name: {evaluation_name}")
    logger.info(f"Model: {tree_search_settings.model.model}")
    logger.info(f"Model Base URL: {tree_search_settings.model.base_url}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Tree Search Settings:")
    logger.info(f"  Max Expansions: {tree_search_settings.max_expansions}")
    logger.info(f"  Max Iterations: {tree_search_settings.max_iterations}")
    logger.info(f"  Min Finished Nodes: {tree_search_settings.min_finished_nodes}")
    logger.info(f"  Provide Feedback: {tree_search_settings.provide_feedback}")
    logger.info(f"  Debate: {tree_search_settings.debate}")
    if tree_search_settings.value_function_model:
        logger.info(
            f"  Value Function Model: {tree_search_settings.value_function_model.model}"
        )
        logger.info(
            f"  Value Function Model Temperature: {tree_search_settings.value_function_model.temperature}"
        )
    logger.info(f"Max Cost: {tree_search_settings.max_cost}")  # TODO: Not used ATM
    logger.info(f"Max iterations: {tree_search_settings.max_iterations}")
    logger.info(f"Number of Workers: {num_workers}")
    logger.info(f"Use Testbed: {use_testbed}")
    logger.info(f"Resolved By: {resolved_by}")
    logger.info(f"Instance IDs: {instance_ids}")

    evaluation = Evaluation(
        settings=tree_search_settings,
        evaluations_dir=evaluations_dir,
        evaluation_name=evaluation_name,
        repo_base_dir=repo_base_dir,
        max_file_context_tokens=16000,
        num_workers=num_workers,
        use_testbed=use_testbed,
    )

    evaluation.run_evaluation(
        resolved_by=resolved_by,
        instance_ids=instance_ids,
    )

    return os.path.join(evaluations_dir, evaluation_name)


# Create a function to ensure the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument(
        "--no_testbed", action="store_true", help="Disable testbed usage"
    )
    parser.add_argument("--debate", action="store_true")
    parser.add_argument("--max_expansions", type=int, default=3)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--max_cost", type=float, default=5.0)
    parser.add_argument("--reward_threshold", type=int, default=None)
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--eval_name", type=str, default=None)
    parser.add_argument("--repo_base_dir", type=str, default=None)
    parser.add_argument("--instance_ids", type=str, nargs="+", default=None)
    parser.add_argument("--sample_first", action="store_true")
    parser.add_argument("--resolved_by", type=int, default=None)
    args = parser.parse_args()

    # Update logging configuration
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.eval_name is not None:
        log_filename = f"run_evaluation_{args.eval_name}_{current_time}.log"
        error_log_filename = f"run_evaluation_{args.eval_name}_{current_time}_error.log"
    elif args.name is not None:
        log_filename = f"run_evaluation_{args.name}_{current_time}.log"
        error_log_filename = f"run_evaluation_{args.name}_{current_time}_error.log"
    else:
        log_filename = f"run_evaluation_{current_time}.log"
        error_log_filename = f"run_evaluation_{current_time}_error.log"

    # Ensure the directory for log files exists
    log_dir = os.path.join(args.eval_dir, "logs")
    ensure_dir(os.path.join(log_dir, log_filename))
    ensure_dir(os.path.join(log_dir, error_log_filename))

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    try:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error creating file handler for logging: {e}")
        print("Continuing without file logging...")

    # Create error file handler which logs error messages
    try:
        error_file_handler = logging.FileHandler(os.path.join(log_dir, error_log_filename))
        error_file_handler.setLevel(logging.ERROR)
        error_file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_file_handler.setFormatter(error_file_formatter)
        logger.addHandler(error_file_handler)
    except Exception as e:
        print(f"Error creating error file handler for logging: {e}")
        print("Continuing without error file logging...")

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARN)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Adjust log levels for specific loggers
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("moatless").setLevel(logging.INFO)
    logging.getLogger("moatless.benchmark.evaluation").setLevel(logging.INFO)
    logging.getLogger("moatless.benchmark.run_evaluation").setLevel(logging.INFO)
    # logging.getLogger("mcts_tree").setLevel(logging.INFO)

    load_dotenv()

    # Create ModelSettings instance
    model_settings = ModelSettings(model=args.model, temperature=args.temp)

    tree_search_settings = TreeSearchSettings(
        max_iterations=args.max_iterations,
        min_finished_transitions=3,
        max_finished_transitions=5,
        reward_threshold=args.reward_threshold,
        provide_feedback=args.feedback,
        debate=args.debate,
        best_first=True,
        model=model_settings
    )

    evaluate_search_and_code(
        evaluation_name=args.eval_name,
        evaluations_dir=args.eval_dir,
        name=args.name,
        repo_base_dir=args.repo_base_dir,
        tree_search_settings=tree_search_settings,
        instance_ids=args.instance_ids,
        date=args.date,
        use_testbed=not args.no_testbed,
        num_workers=args.num_workers,
        best_first=not args.sample_first,
        resolved_by=args.resolved_by,
    )
