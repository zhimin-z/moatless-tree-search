from moatless.actions.find_function import FindFunction
from moatless.actions.find_class import FindClass
from moatless.benchmark.swebench import create_workspace
from moatless.benchmark.utils import get_moatless_instance


def test_find_function_init_method():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    workspace = create_workspace(instance)

    action = FindFunction(
        scratch_pad="",
        class_name="ManagementUtility",
        function_name="__init__",
    )
    action.set_workspace(workspace)

    file_context = workspace.create_file_context()
    message = action.execute(file_context)
    print(message)
    assert len(file_context.files) == 1
    assert "ManagementUtility.__init__" in file_context.files[0].span_ids


def test_find_function():
    instance_id = "django__django-14855"
    instance = get_moatless_instance(instance_id)
    workspace = create_workspace(instance)

    action = FindFunction(
        scratch_pad="",
        function_name="cached_eval",
        file_pattern="**/*.py",
    )
    action.set_workspace(workspace)

    file_context = workspace.create_file_context()
    message = action.execute(file_context)
