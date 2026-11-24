"""Opt-in RelayVSR decoder; old ComfyUI builds remain importable."""
import folder_paths


class RayRelayVSRDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "images": ("IMAGE",),
            "vae_name": (folder_paths.get_filename_list("vae"),),
            "detail": ("FLOAT", {"default": 1.0, "min": 0., "max": 1., "step": 0.01}),
            "worker_count": ([1, 2], {"default": 2}),
            "keep_workers": ("BOOLEAN", {"default": False,
                "tooltip": "Retain GPU decoder weights for reuse; unused scratch is always released. No extra processes."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Raylight/RelayVSR"
    DESCRIPTION = "Exact FP32 RelayVSR Flash windows on one or two GPUs. Keep native RelayVSR sampling."

    def decode(self, samples, images, vae_name, detail, worker_count, keep_workers, ray_actors=None,
               execution_mode="local_threads"):
        import comfy.model_management as mm
        from .local import run_local_flash
        if execution_mode != "local_threads" or ray_actors is not None:
            raise ValueError("Ray actor RelayVSR decoding is an unshipped experiment; use local_threads without ray_actors")
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        # Sampling has finished. Do not keep the native sampler's GPU weights
        # resident alongside Flash actors. CPU workflow caches are not destroyed.
        def prepare():
            mm.unload_all_models()
            mm.soft_empty_cache()
        result = run_local_flash(path, samples["samples"], images, detail,
            worker_count=int(worker_count), keep_workers=keep_workers,
            interrupt_check=mm.throw_exception_if_processing_interrupted, prepare=prepare)
        return (result,)


NODE_CLASS_MAPPINGS = {"RayRelayVSRDecode": RayRelayVSRDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"RayRelayVSRDecode": "RelayVSR Flash Decode (Ray)"}
