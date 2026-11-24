"""Official RelayFlash loading and explicitly owned VAE lifecycle."""
import math
from pathlib import Path

import torch

from .planning import plan_relay_windows


def validate_relay_inputs(latents, images, scale, detail=1.0):
    if scale not in (2, 4):
        raise ValueError("RelayVSR scale must be x2 or x4")
    if not isinstance(detail, (int, float)) or not math.isfinite(detail) or not 0 <= detail <= 1:
        raise ValueError("RelayVSR detail must be finite and between 0 and 1")
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("RelayVSR images must be F,H,W,3")
    if not images.is_floating_point() or min(images.shape[:3]) < 1:
        raise ValueError("RelayVSR images must contain nonempty floating-point frames")
    jobs = plan_relay_windows(int(images.shape[0]))
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5 or tuple(latents.shape[:2]) != (1, 48):
        raise ValueError("RelayVSR expects one B=1,C=48 video latent")
    stride = (16 // scale) * 2
    spatial = tuple(math.ceil(int(s) / stride)*2 for s in images.shape[1:3])
    if tuple(latents.shape[2:]) != (jobs[-1].latent_stop, *spatial):
        raise ValueError("RelayVSR latent count/spatial size does not match images and VAE scale")
    return jobs


class RelayFlashDecoder:
    def __init__(self, vae_path, dtype="fp32", device=None):
        if dtype != "fp32":
            raise ValueError("Only the validated FP32 RelayFlash decoder is enabled")
        self.vae_path = str(Path(vae_path).resolve())
        self.device = device
        self.vae = None

    def load(self):
        if self.vae is None:
            try:
                from comfy.ldm.wan.model_vsr import RelayFlash, RELAY_KEYFRAME_INTERVAL
            except ImportError as error:
                raise RuntimeError("This ComfyUI does not include native RelayVSR Flash support") from error
            if RELAY_KEYFRAME_INTERVAL != 15:
                raise RuntimeError("Unvalidated RelayVSR keyframe interval; expected 15")
            import comfy.sd
            import comfy.utils
            sd, metadata = comfy.utils.load_torch_file(self.vae_path, return_metadata=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=torch.float32, device=self.device)
            vae.throw_exception_if_invalid()
            if not isinstance(vae.first_stage_model, RelayFlash):
                raise ValueError("Selected VAE is not an official RelayVSR Flash decoder")
            self.vae = vae
        return self.vae

    @torch.inference_mode()
    def decode_window(self, latents, images, detail=1.0):
        vae = self.load()
        validate_relay_inputs(latents, images, vae.first_stage_model.upscale_factor, detail)
        decoded = vae.decode(latents, vae_options={"lq_images": images, "detail": detail})
        return decoded.reshape(-1, *decoded.shape[-3:]).detach().cpu().float()

    def close(self):
        if self.vae is None:
            return
        import comfy.model_management as mm
        patcher = self.vae.patcher
        for loaded in list(mm.current_loaded_models):
            if loaded.model is patcher:
                loaded.model_unload()
                mm.current_loaded_models.remove(loaded)
        # The direct local path is not in current_loaded_models. Native patchers
        # can create model -> current_patcher -> model cycles; detach offloads
        # owned GPU weights, cleanup breaks the back-reference deterministically.
        patcher.detach()
        patcher.cleanup()
        self.vae = None
        mm.soft_empty_cache()



