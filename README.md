# Moatless-Tree-Search 

Note: The original development code can be found at [https://github.com/a-antoniades/swe-planner](https://github.com/a-antoniades/swe-planner). It is only intended for reproducing the results in the paper. This is a clean refactor with a modular design, which will be extended and actively maintained.

<div align="center">

[![License](https://img.shields.io/badge/LICENSE-APACHE_LICENSE_2.0-yellow?style=flat-square&labelColor=lightgrey)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2408.08435-B31B1B?style=flat-square)](https://arxiv.org/html/2410.20285v1)
[![Streamlit](https://img.shields.io/badge/STREAMLIT-7C4DFF?style=flat-square)](https://streamlit.moatless.ai/)
[![Twitter](https://img.shields.io/badge/TWITTER-00ACEE?style=flat-square)](https://twitter.com/your-handle)
[![YouTube](https://img.shields.io/badge/YOUTUBE-FF0000?style=flat-square)](https://www.youtube.com/watch?v=VcEHX_TNDgQ)
</div>

<div align="center">
  <a href="assets/method.pdf" target="_blank">
    <img src="./assets/method.png" alt="Method Diagram" width="75%">
  </a>

  <p><strong>Figure 1:</strong> Overview of SWE-Search showing the tree search process, where states (nodes) and actions (edges) are evaluated using contextual information and value function feedback to guide expansion.</p>
</div>

## Installation

1. Clone the repository and create a conda environment:

```bash
git clone https://github.com/aorwall/moatless-tree-search.git
cd moatless-tree-search
conda create -n moatless python=3.11
conda activate moatless
```

2. Install Poetry (if not already installed):

```bash
conda install -c conda-forge poetry
```

3. Install dependencies using Poetry:
    
```bash
poetry install
```

## Environment Setup

Before running the evaluation, you'll need to set up your environment variables. Add these to your `~/.bashrc` (bash) or `~/.zshrc` (zsh):

```bash
# Base URL for custom LLM API service (optional)
export CUSTOM_LLM_API_BASE="<your-base-url>"
export CUSTOM_LLM_API_KEY="<your-key>"

# API keys for various LLM providers
export OPENAI_API_KEY="<your-key>"
export ANTHROPIC_API_KEY="<your-key>"
export HUGGINGFACE_API_KEY="<your-key>"
export DEEPSEEK_API_KEY="<your-key>"

# API Keys for Voyage Embeddings
export VOYAGE_API_KEY="<your-key>"
export INDEX_STORE_DIR="<your-index-store-dir>" # default: /tmp/index_store

# Testbed configuration for evaluation environment
export TESTBED_API_KEY="<your-key>"
export TESTBED_BASE_URL="<your-base-url>"
```


## Streamlit

To launch the Streamlit app, run:

```bash
streamlit run streamlit_app.py
```

The following badges are used to indicate the status of a node:

| Badge | Shape | Color | Description |
|-------|-------|-------|-------------|
| ⭐ | Star | Green | Node is marked as resolved |
| ❌ | X | Red | Node is either unresolved or has warnings |
| 🟢 | Circle | Green | Found correct spans in the right context |
| 🟡 | Circle | Yellow | Either:<br>• Found files but not spans<br>• Found spans but in wrong files<br>• Found right files (patch status) |

## Evaluation

To run the evaluation script, use the following command:

```bash
python ./moatless/benchmark/run_evaluation.py \
        --model "gpt-4o-mini-2024-07-18" \
        --repo_base_dir $MOATLESS_REPO_BASE_DIR \
        --eval_dir $MOATLESS_EVAL_DIR \
        --eval_dir "./evaluations" \
        --eval_name mts \
        --temp 0.7 \
        --num_workers 1 \
        --feedback \
        --max_iterations 100 \
        --max_expansions 5
```

You can optionally set the `--instance_id` to evaluate on a specific instance or a list of instances.

## Description of the Flow
The search algorithm operates in a loop, following these main steps to explore and evaluate possible actions:

1. **Selection:** Choose the next `Node` to expand using a `Selector` strategy. The selector evaluates the available nodes (expandable descendants) and selects the most promising one based on predefined criteria.

2. **Expansion:** Create a new child `Node` or select an existing unexecuted child.

3. **Simulation:**
   * **Action Generation and Execution:** Use the `Agent` to generate and execute an action for the `Node`:
     - Generate possible actions for the node
     - Create a system prompt and messages based on the node's history
     - Use a `CompletionModel` to generate action arguments
     - Execute the chosen action, updating the node's `FileContext` and creating an `Observation`
   * **Reward Evaluation:** If a `ValueFunction` is defined, evaluate the outcome of the action execution, assigning a reward to the `Node`.

4. **Backpropagation:** Propagate the obtained reward back up the tree, updating the value estimates and visit counts of ancestor nodes.

When the search process finishes (based on predefined stopping criteria), the algorithm determines the best solution found using a `Discriminator`, which assesses the nodes based on their rewards and other factors.

### Node
   
```python
class Node(BaseModel):
    node_id: int = Field(..., description="The unique identifier of the node")
    parent: Optional['Node'] = Field(None, description="The parent node")
    children: List['Node'] = Field(default_factory=list, description="The child nodes")
    is_duplicate: bool = Field(False, description="Flag to indicate if the node is a duplicate")
    action: Optional[ActionArguments] = Field(None, description="The action associated with the node")
    observation: Optional[Observation] = Field(None, description="The observation from the executed action")
    reward: Optional[Reward] = Field(None, description="The reward of the node")
    visits: int = Field(0, description="The number of times the node has been visited")
    value: float = Field(0.0, description="The total value (reward) of the node")
    file_context: Optional[FileContext] = Field(None, description="The file context state associated with the node")
    message: Optional[str] = Field(None, description="The message associated with the node")
    feedback: Optional[str] = Field(None, description="Feedback provided to the node")
    completions: Dict[str, Completion] = Field(default_factory=dict, description="The completions used in this node")
``` 

### Selector

```python
class Selector(ABC):
    @abstractmethod
    def select(self, nodes: List[Node]) -> Optional[Node]:
        """
        Selects the next node to expand from a list of candidate nodes.

        Args:
            nodes (List[Node]): A list of candidate nodes to select from.

        Returns:
            Optional[Node]: The selected node, or None if no suitable node is found.
        """
        pass
```

### Agent
Responsible for generating and executing actions:

```python
class Agent(BaseModel):
    def run(self, node: Node):
        self._generate_action(node)
        self._execute_action(node)
```

### Action
Defines the structure and execution logic for different types of actions:

```python
class Action(BaseModel, ABC):
    args_schema: Type[ActionArguments]

    def execute(self, args: ActionArguments, file_context: FileContext) -> Observation:
        message = self._execute(file_context=file_context)
        return Observation.create(message)
```

### ActionArguments
Defines the arguments required for each action type:

```python
class ActionArguments(OpenAISchema, ABC):
    scratch_pad: str = Field(..., description="Your reasoning for the action.")
```

### Observation
Represents the result of an action execution:

```python
class Observation(BaseModel):
    message: str
    extra: Optional[str] = None
    terminal: bool = False
    expect_correction: bool = False
    properties: Optional[Dict[str, Any]] = None
    execution_completion: Optional[Completion] = None
```

### ValueFunction

```python
class ValueFunction(ABC):
    @abstractmethod
    def get_reward(self, node: Node) -> Reward:
        """
        Evaluates the node's outcome and assigns a reward.

        Args:
            node (Node): The node to evaluate.

        Returns:
            Reward: The calculated reward
        """
        pass
``` 

#### Reward

```python
class Reward(OpenAISchema):
    explanation: Optional[str] = Field(None, description="An explanation and the reasoning behind your decision.")
    feedback: Optional[str] = Field(None, description="Feedback to the alternative branch.")
    value: int = Field(..., description="As ingle integer value between -100 and 100 based on your confidence in the correctness of the action and its likelihood of resolving the issue")
``` 

### FeedbackGenerator

```python

class FeedbackGenerator(ABC):
    @abstractmethod
    def generate_feedback(self, node: Node) -> str:
        """
        Generates feedback for a given node to inform the next action.

        Args:
            node (Node): The node for which to generate feedback.

        Returns:
            str: The feedback message to be used in action building.
        """
        pass
``` 

### Discriminator

```python
class Discriminator(ABC):
    @abstractmethod
    def select(self, nodes: List['Node']) -> Optional['Node']:
        """
        Selects the best node from a list of nodes based on specific criteria.

        Args:
            nodes (List[Node]): A list of nodes to evaluate.

        Returns:
            Optional[Node]: The node deemed the best, or None if no suitable node is found.
        """
        pass
``` 
