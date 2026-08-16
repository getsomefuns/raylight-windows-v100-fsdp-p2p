import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "tests" / "windows_p2p_fsdp_probe.py"


def _tree():
    return ast.parse(PROBE_PATH.read_text(encoding="utf-8"))


def _actor_method(name):
    actor = next(node for node in _tree().body if isinstance(node, ast.ClassDef) and node.name == "FSDPProbeActor")
    return next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_quantized_probe_can_exercise_runtime_cpu_offload():
    method = _actor_method("run_quantized_forward")
    source = ast.unparse(method)
    argument_names = [argument.arg for argument in method.args.args]

    assert "cpu_offload" in argument_names
    assert "CPUOffloadPolicy(pin_memory=False)" in source
    assert "cpu_offload=cpu_offload" in source
    assert "local_qdata_device" in source


def test_probe_cli_exposes_cpu_offload_and_uses_the_sibling_comfyui():
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert 'parser.add_argument("--cpu-offload", action="store_true")' in source
    assert "actor.run_quantized_forward.remote(args.cpu_offload)" in source
    assert 'REPO_ROOT.parent / "ComfyUI"' in source
