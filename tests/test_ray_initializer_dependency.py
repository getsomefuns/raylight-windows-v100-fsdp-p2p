import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "src" / "raylight" / "nodes.py"


def _class_node(name):
    tree = ast.parse(NODES_PATH.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method_node(class_definition, name):
    return next(node for node in class_definition.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_ray_initializer_exposes_a_pure_execution_dependency_input():
    initializer = _class_node("RayInitializer")
    input_types = ast.unparse(_method_node(initializer, "INPUT_TYPES"))
    spawn_actor = _method_node(initializer, "spawn_actor")
    argument_names = [argument.arg for argument in spawn_actor.args.args]

    assert "'optional': {'wait_for': (any_type" in input_types
    assert "wait_for" in argument_names


def test_advanced_initializer_keeps_the_same_execution_dependency_input():
    initializer = _class_node("RayInitializerAdvanced")
    input_types = ast.unparse(_method_node(initializer, "INPUT_TYPES"))

    assert "'wait_for': (any_type" in input_types
