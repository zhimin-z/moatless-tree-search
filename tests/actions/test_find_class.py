from moatless.actions.find_class import FindClass
from moatless.benchmark.swebench import create_workspace
from moatless.benchmark.utils import get_moatless_instance


def test_find_class__too_large():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    workspace = create_workspace(instance)

    action = FindClass(
        scratch_pad="",
        class_name="ManagementUtility",
    )
    action.set_workspace(workspace)

    file_context = workspace.create_file_context()
    output = action.execute(file_context)
    print(output.extra)
    assert len(file_context.files) == 0
    assert output.extra.startswith("The class is too large.")
    assert "ManagementUtility" in output.extra
    assert output.message == "Found 1 classes."


def test_find_class__wrong_file_pattern():
    instance_id = "django__django-13658"
    instance = get_moatless_instance(instance_id)
    workspace = create_workspace(instance)

    action = FindClass(
        scratch_pad="", class_name="ManagementUtility", file_pattern="**/foo/**"
    )
    action.set_workspace(workspace)

    file_context = workspace.create_file_context()
    output = action.execute(file_context)
    print(output.extra)
    assert len(file_context.files) == 0
    assert (
        output.message
        == "No files found for file pattern **/foo/**. But found 1 alternative suggestions."
    )
    assert "ManagementUtility" in output.extra
