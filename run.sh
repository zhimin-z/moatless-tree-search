
# models
# claude-3-5-sonnet-20241022
# gpt-4o-mini-2024-07-18

MODEL="claude-3-5-sonnet-20241022"
PYTHONPATH=$PYTHONPATH:$(pwd)
CWD=$(pwd)
echo $CWD

python ./moatless/benchmark/run_evaluation.py \
        --model $MODEL \
        --repo_base_dir "$CWD/repos" \
        --eval_dir "$CWD/evaluations" \
        --eval_name debug/$MODEL \
        --temp 0.7 \
        --num_workers 1 \
        --feedback \
        --instance_id django__django-15252 \
        --max_iterations 50 \
        --max_expansions 5