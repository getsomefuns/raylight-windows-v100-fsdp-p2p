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
            "execution_mode": (["local_threads", "ray_actors"], {"default": "local_threads"}),
            "keep_workers": ("BOOLEAN", {"default": False,
                "tooltip": "Retain only this node's dedicated Flash pool for warm reuse."}),
        }, "optional": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Raylight/RelayVSR"
    DESCRIPTION = "Exact FP32 RelayVSR Flash windows on one or two GPUs. Keep native RelayVSR sampling."

    def decode(self, samples, images, vae_name, detail, worker_count, keep_workers, ray_actors=None,
               execution_mode="local_threads"):
        import comfy.model_management as mm
        from .runtime import run_parallel_flash, release_relayvsr_pool
        from .local import run_local_flash, release_local_pool
        if execution_mode not in ("local_threads", "ray_actors"):
            raise ValueError("Unknown RelayVSR execution mode")
        if execution_mode == "local_threads" and ray_actors is not None:
            raise ValueError("local_threads cannot borrow ray_actors; select ray_actors mode")
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        # Sampling has finished. Do not keep the native sampler's GPU weights
        # resident alongside Flash actors. CPU workflow caches are not destroyed.
        mm.unload_all_models()
        mm.soft_empty_cache()
        workers = () if ray_actors is None else ray_actors["workers"]
        if execution_mode == "local_threads":
            release_relayvsr_pool()
            execute = run_local_flash
        else:
            release_local_pool()
            execute = run_parallel_flash
        result = execute(path, samples["samples"], images, detail,
            worker_count=int(worker_count), keep_workers=keep_workers,
            borrowed_workers=workers, interrupt_check=mm.throw_exception_if_processing_interrupted)
        return (result,)


NODE_CLASS_MAPPINGS = {"RayRelayVSRDecode": RayRelayVSRDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"RayRelayVSRDecode": "RelayVSR Flash Decode (Ray)"}
