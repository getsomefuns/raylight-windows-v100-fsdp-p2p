from types import SimpleNamespace

import torch

from raylight.comfy_dist.fsdp_utils import configure_minimax_h3_forward_prefetch


class FakeFSDP(torch.nn.Module):
    def __init__(self, gathered_bytes):
        super().__init__()
        local_bytes = gathered_bytes // 2
        local_elements = max(1, local_bytes // 2)
        inputs = [torch.empty(local_elements, dtype=torch.float16)]
        param = SimpleNamespace(all_gather_inputs=inputs)
        group = SimpleNamespace(
            fsdp_params=[param],
            mesh_info=SimpleNamespace(mesh=SimpleNamespace(size=lambda: 2)),
            unshard_async_op=True,
        )
        self.state = SimpleNamespace(_fsdp_param_group=group)
        self.prefetch_targets = []
        self.async_values = []

    def _get_fsdp_state(self):
        return self.state

    def set_modules_to_forward_prefetch(self, modules):
        self.prefetch_targets = list(modules)

    def _set_unshard_async_op(self, value):
        self.async_values.append(value)
        self.state._fsdp_param_group.unshard_async_op = value


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = FakeFSDP(110 * 1024**2)
        self.q_norm = FakeFSDP(1 * 1024**2)
        self.k_norm = FakeFSDP(1 * 1024**2)
        self.out_proj = FakeFSDP(56 * 1024**2)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FakeFSDP(147 * 1024**2)
        self.fc2 = FakeFSDP(74 * 1024**2)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = FakeFSDP(1 * 1024**2)
        self.norm2 = FakeFSDP(1 * 1024**2)
        self.attn = Attention()
        self.mlp = MLP()


class MiniMaxH3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])


def test_minimax_prefetch_configures_only_targets_within_budget():
    model = MiniMaxH3Model()

    result = configure_minimax_h3_forward_prefetch(
        model,
        max_prefetch_bytes=128 * 1024**2,
        fsdp_module_type=FakeFSDP,
    )

    assert result == {
        "configured": 6,
        "skipped": 2,
        "max_configured_bytes": 110 * 1024**2,
    }
    for block in model.blocks:
        assert block.norm1.prefetch_targets == [block.attn.qkv_proj]
        assert block.attn.k_norm.prefetch_targets == [block.attn.out_proj]
        assert block.norm2.prefetch_targets == []
        assert block.mlp.fc1.prefetch_targets == [block.mlp.fc2]
        assert block.attn.qkv_proj.async_values == [False]
        assert block.attn.out_proj.async_values == [False]
        assert block.mlp.fc2.async_values == [False]
        assert block.mlp.fc1.async_values == []


def test_minimax_prefetch_is_inactive_for_zero_budget_or_other_models():
    model = MiniMaxH3Model()
    assert configure_minimax_h3_forward_prefetch(
        model,
        max_prefetch_bytes=0,
        fsdp_module_type=FakeFSDP,
    ) == {"configured": 0, "skipped": 0, "max_configured_bytes": 0}

    assert configure_minimax_h3_forward_prefetch(
        torch.nn.Module(),
        max_prefetch_bytes=128 * 1024**2,
        fsdp_module_type=FakeFSDP,
    ) == {"configured": 0, "skipped": 0, "max_configured_bytes": 0}
