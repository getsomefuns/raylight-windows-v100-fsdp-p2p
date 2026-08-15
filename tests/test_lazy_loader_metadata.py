import types
import unittest
from unittest import mock


class LazyLoaderMetadataTests(unittest.TestCase):
    def test_metadata_from_safetensors_is_forwarded_to_comfy(self):
        from raylight.comfy_dist import sd as sd_module

        state_dict = {"weight": object()}
        metadata = {"format": "pt", "modelspec.architecture": "ltxv"}
        lazy_state_dict = {"lazy_weight": object()}
        loaded_model = types.SimpleNamespace()

        lazy_tensor_module = types.ModuleType(
            "raylight.expansion.comfyui_lazytensors.lazy_tensor"
        )
        lazy_tensor_module.wrap_state_dict_lazy = mock.Mock(
            return_value=lazy_state_dict
        )
        ops_module = types.ModuleType(
            "raylight.expansion.comfyui_lazytensors.ops"
        )
        ops_module.SafetensorOps = object()

        with (
            mock.patch.object(
                sd_module,
                "load_safetensors_mmap_with_metadata",
                return_value=(state_dict, metadata),
            ),
            mock.patch.dict(
                "sys.modules",
                {
                    lazy_tensor_module.__name__: lazy_tensor_module,
                    ops_module.__name__: ops_module,
                },
            ),
            mock.patch.object(
                sd_module.comfy.sd,
                "load_diffusion_model_state_dict",
                return_value=loaded_model,
            ) as load_model,
        ):
            result = sd_module.lazy_load_diffusion_model(
                "ltx-2.3-fp8.safetensors",
                model_options={"dtype": "fp8", "use_mmap": True},
            )

        self.assertIs(result, loaded_model)
        self.assertIs(result.mmap_cache, state_dict)
        load_model.assert_called_once()
        _, kwargs = load_model.call_args
        self.assertIs(kwargs["metadata"], metadata)
        self.assertIs(
            kwargs["model_options"]["custom_operations"],
            ops_module.SafetensorOps,
        )
        self.assertNotIn("dtype", kwargs["model_options"])


if __name__ == "__main__":
    unittest.main()
