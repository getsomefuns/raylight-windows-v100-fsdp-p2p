"""Exercise actual native clone(), not a stand-in constructor signature."""
import inspect

import pytest
import torch

from comfy.model_patcher import ModelPatcher
from raylight.comfy_dist.model_patcher import FSDPModelPatcher, PipefusionModelPatcher

pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")


@pytest.mark.parametrize("kind", [FSDPModelPatcher, PipefusionModelPatcher])
@pytest.mark.parametrize("fast_disk", [False, True])
def test_custom_patcher_preserves_native_disk_policy_through_clone(kind, fast_disk):
    model = torch.nn.Linear(3, 2)
    supports = "fast_disk" in inspect.signature(ModelPatcher.__init__).parameters
    if fast_disk and not supports:
        with pytest.raises(ValueError, match="fast_disk"):
            kind(model, torch.device("cpu"), torch.device("cpu"), fast_disk=True)
        return
    patcher = kind(model, torch.device("cpu"), torch.device("cpu"), fast_disk=fast_disk)
    patcher.add_patches({"weight": ("diff", (torch.ones_like(model.weight),))}, 0.75)
    clone = patcher.clone()
    assert type(clone) is kind
    assert clone.model is model
    assert clone.patches == patcher.patches
    assert getattr(clone, "fast_disk", False) is fast_disk
    if kind is FSDPModelPatcher:
        assert clone.rank == patcher.rank and clone.fsdp_state_dict is patcher.fsdp_state_dict
