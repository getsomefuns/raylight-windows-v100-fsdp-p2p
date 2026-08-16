from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from comfy.quant_ops import QuantizedTensor, get_layout_class
from raylight.comfy_dist.kitchen_patches.fp8 import (
    install_fp8_patches,
    restore_fp8_patches,
)


class FP8FSDPGatherShapeTests(unittest.TestCase):
    def tearDown(self):
        restore_fp8_patches()

    def test_post_all_gather_restores_global_logical_shape(self):
        layout_name = "TensorCoreFP8E4M3Layout"
        layout = get_layout_class(layout_name)
        scale = torch.tensor(0.25, dtype=torch.float32)
        local_qdata = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(
            torch.float8_e4m3fn
        )
        local = QuantizedTensor(
            local_qdata,
            layout_name,
            layout.Params(
                scale=scale,
                orig_dtype=torch.bfloat16,
                orig_shape=(2, 4),
            ),
        )
        peer_qdata = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + 8).to(
            torch.float8_e4m3fn
        )
        gathered_qdata = torch.cat((local_qdata, peer_qdata), dim=0)
        install_fp8_patches()

        gathered, _inner = local.fsdp_post_all_gather(
            (gathered_qdata,), (scale,), torch.bfloat16
        )

        self.assertEqual(tuple(gathered.shape), (4, 4))
        self.assertEqual(tuple(gathered._params.orig_shape), (4, 4))
        self.assertEqual(tuple(gathered._qdata.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()