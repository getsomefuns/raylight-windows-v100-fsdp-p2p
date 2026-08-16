import unittest
from types import SimpleNamespace

import torch

from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops


class QuantOpsWrapperScopeTests(unittest.TestCase):
    def test_non_fsdp_overwrite_cast_does_not_depend_on_fsdp_import_scope(self):
        worker = SimpleNamespace(
            parallel_dict={"is_fsdp": False, "is_quant": False},
            overwrite_cast_dtype=torch.float32,
        )

        @patch_temp_fix_ck_ops
        def sample(self):
            return "ok"

        self.assertEqual(sample(worker), "ok")


if __name__ == "__main__":
    unittest.main()