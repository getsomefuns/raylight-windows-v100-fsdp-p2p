"""Ray actor probe for raw CUDA IPC metadata and NVLink data plane."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import time

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"

import ray


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/distributed_worker/windows_p2p.py"


@ray.remote(num_gpus=1)
class P2PActor:
    def __init__(self, rank, group_name, capacity_bytes):
        import sys
        import torch

        spec = importlib.util.spec_from_file_location(f"raylight_windows_p2p_ray_{rank}", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.cuda.set_device(0)
        self.rank = rank
        self.module = module
        self.control = module.WindowsSpinControl(group_name, rank)
        self.endpoint = module.CudaP2PAllToAll(rank, capacity_bytes, self.control, timeout_seconds=10)

    def metadata(self):
        return self.endpoint.local_ipc_metadata()

    def connect(self, peer_metadata):
        self.endpoint.connect_ipc_metadata(peer_metadata)
        return True

    def run(self, iterations, size_bytes):
        import torch

        elements = size_bytes // 4
        half = elements // 2
        source = torch.empty(elements, dtype=torch.float32, device="cuda:0")
        source[:half].fill_(self.rank * 100 + 1)
        source[half:].fill_(self.rank * 100 + 2)
        output = torch.empty_like(source)
        for _ in range(3):
            self.endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            self.endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = (1, 101) if self.rank == 0 else (2, 102)
        observed = (int(output[0].item()), int(output[half].item()))
        return {
            "rank": self.rank,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "values": observed,
            "expected": expected,
            "elapsed_seconds": elapsed,
            "remote_gib_s": (size_bytes / 2 / 2**30) * iterations / elapsed,
        }

    def run_all_gather(self, iterations, shard_bytes):
        import torch

        elements = shard_bytes // 4
        source = torch.full(
            (elements,),
            self.rank + 1,
            dtype=torch.float32,
            device="cuda:0",
        )
        output = torch.empty(elements * 2, dtype=torch.float32, device="cuda:0")
        for _ in range(3):
            self.endpoint.all_gather_into_tensor(output, source, async_op=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            self.endpoint.all_gather_into_tensor(output, source, async_op=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        observed = (int(output[0].item()), int(output[elements].item()))
        return {
            "rank": self.rank,
            "values": observed,
            "expected": (1, 2),
            "elapsed_seconds": elapsed,
            "remote_gib_s": (shard_bytes / 2**30) * iterations / elapsed,
        }

    def run_all_gather_async(self, shard_bytes):
        import torch
        import torch.distributed as dist

        elements = shard_bytes // 4
        source = torch.full(
            (elements,),
            self.rank + 1,
            dtype=torch.float32,
            device="cuda:0",
        )
        output = torch.empty(elements * 2, dtype=torch.float32, device="cuda:0")
        work = self.endpoint.all_gather_into_tensor(output, source, async_op=True)
        is_work = isinstance(work, dist.Work)
        wait_result = work.wait()
        torch.cuda.synchronize()
        return {
            "rank": self.rank,
            "is_work": is_work,
            "wait_result": wait_result,
            "values": (int(output[0].item()), int(output[elements].item())),
            "expected": (1, 2),
        }

    def run_chunked_all_gather(self, shard_bytes, capacity_bytes, async_op):
        import torch

        elements = shard_bytes // 4
        boundary = capacity_bytes // 4
        source = torch.full(
            (elements,),
            self.rank * 100 + 1,
            dtype=torch.float32,
            device="cuda:0",
        )
        source[boundary - 1] = self.rank * 100 + 2
        source[boundary] = self.rank * 100 + 3
        source[-1] = self.rank * 100 + 4
        output = torch.empty(elements * 2, dtype=torch.float32, device="cuda:0")
        operation_id_before = self.endpoint._operation_id
        torch.cuda.synchronize()
        started = time.perf_counter()
        work = self.endpoint.all_gather_into_tensor(output, source, async_op=async_op)
        operation_id_after = self.endpoint._operation_id
        if async_op:
            work.wait()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        positions = (0, boundary - 1, boundary, elements - 1)
        observed = tuple(
            int(output[rank_offset * elements + position].item())
            for rank_offset in range(2)
            for position in positions
        )
        return {
            "rank": self.rank,
            "async_op": async_op,
            "values": observed,
            "expected": (1, 2, 3, 4, 101, 102, 103, 104),
            "operation_ids_consumed": operation_id_after - operation_id_before,
            "expected_chunks": len(self.module.chunk_ranges(shard_bytes, capacity_bytes)),
            "elapsed_seconds": elapsed,
        }


    def run_dtype_tail(
        self, dtype_name, shard_bytes, capacity_bytes, iterations, async_op
    ):
        import torch

        dtype = getattr(torch, dtype_name)
        element_size = torch.empty((), dtype=dtype).element_size()
        if shard_bytes % element_size:
            raise ValueError((dtype_name, shard_bytes, element_size))
        elements = shard_bytes // element_size
        source = torch.empty(elements, dtype=dtype, device="cuda:0")
        source_bytes = source.view(torch.uint8).reshape(-1)
        base = self.rank * 16 + 1
        source_bytes.fill_(base)
        source_bytes[capacity_bytes - 1] = base + 1
        source_bytes[capacity_bytes] = base + 2
        source_bytes[-1] = base + 3
        output = torch.empty(elements * 2, dtype=dtype, device="cuda:0")

        operation_id_before = self.endpoint._operation_id
        for _ in range(iterations):
            work = self.endpoint.all_gather_into_tensor(
                output, source, async_op=async_op
            )
            if async_op:
                work.wait()
        torch.cuda.synchronize()
        operation_id_after = self.endpoint._operation_id

        output_bytes = output.view(torch.uint8).reshape(-1)
        positions = (0, capacity_bytes - 1, capacity_bytes, shard_bytes - 1)
        observed = tuple(
            int(output_bytes[rank_offset * shard_bytes + position].item())
            for rank_offset in range(2)
            for position in positions
        )
        expected_chunks = len(self.module.chunk_ranges(shard_bytes, capacity_bytes))
        return {
            "rank": self.rank,
            "dtype": dtype_name,
            "async_op": async_op,
            "values": observed,
            "expected": (1, 2, 3, 4, 17, 18, 19, 20),
            "operation_ids_consumed": operation_id_after - operation_id_before,
            "expected_operation_ids": expected_chunks * iterations,
            "iterations": iterations,
        }


    def close(self):
        self.endpoint.close()
        self.control.close()


if __name__ == "__main__":
    size_bytes = 115_343_360
    capacity_bytes = 128 * 1024 * 1024
    iterations = 100
    minimum_gib_s = float(os.environ.get("RAYLIGHT_WINDOWS_P2P_MIN_GIB_S", "50"))
    ray.init(num_gpus=2, include_dashboard=False, ignore_reinit_error=True)
    actors = []
    try:
        group_name = f"ray_{os.getpid()}_{time.time_ns()}"
        actors = [P2PActor.remote(rank, group_name, capacity_bytes) for rank in range(2)]
        metadata = ray.get([actor.metadata.remote() for actor in actors])
        ray.get([actors[rank].connect.remote(metadata[1 - rank]) for rank in range(2)])
        results = ray.get([actor.run.remote(iterations, size_bytes) for actor in actors], timeout=120)
        for result in results:
            print(result)
            if result["values"] != result["expected"]:
                raise RuntimeError(result)
            if result["remote_gib_s"] < minimum_gib_s:
                raise RuntimeError(
                    f"rank {result['rank']} P2P bandwidth {result['remote_gib_s']:.2f} GiB/s "
                    f"is below the required {minimum_gib_s:.2f} GiB/s"
                )
        shard_bytes = size_bytes // 2
        gather_results = ray.get(
            [actor.run_all_gather.remote(iterations, shard_bytes) for actor in actors],
            timeout=120,
        )
        for result in gather_results:
            print({"all_gather": result})
            if result["values"] != result["expected"]:
                raise RuntimeError(result)

        async_results = ray.get(
            [actor.run_all_gather_async.remote(shard_bytes) for actor in actors],
            timeout=120,
        )
        for result in async_results:
            print({"all_gather_async": result})
            if not result["is_work"] or not result["wait_result"]:
                raise RuntimeError(result)
            if result["values"] != result["expected"]:
                raise RuntimeError(result)

        for benchmark_mib in (64, 128, 256, 384):
            benchmark_shard_bytes = benchmark_mib * 1024 * 1024
            benchmark_results = ray.get(
                [
                    actor.run_all_gather.remote(20, benchmark_shard_bytes)
                    for actor in actors
                ],
                timeout=120,
            )
            for result in benchmark_results:
                print({"all_gather_mib": benchmark_mib, **result})
                if result["values"] != result["expected"]:
                    raise RuntimeError(result)

        large_shard_bytes = 256 * 1024 * 1024
        for async_op in (False, True):
            chunked_results = ray.get(
                [
                    actor.run_chunked_all_gather.remote(
                        large_shard_bytes, capacity_bytes, async_op
                    )
                    for actor in actors
                ],
                timeout=120,
            )
            for result in chunked_results:
                print({"chunked_all_gather": result})
                if result["values"] != result["expected"]:
                    raise RuntimeError(result)
                if result["operation_ids_consumed"] != result["expected_chunks"]:
                    raise RuntimeError(result)

        tail_shard_bytes = capacity_bytes + 4096
        for dtype_name in ("float32", "float16", "bfloat16", "uint8"):
            for async_op in (False, True):
                dtype_results = ray.get(
                    [
                        actor.run_dtype_tail.remote(
                            dtype_name,
                            tail_shard_bytes,
                            capacity_bytes,
                            20,
                            async_op,
                        )
                        for actor in actors
                    ],
                    timeout=120,
                )
                for result in dtype_results:
                    print({"dtype_tail_all_gather": result})
                    if result["values"] != result["expected"]:
                        raise RuntimeError(result)
                    if (
                        result["operation_ids_consumed"]
                        != result["expected_operation_ids"]
                    ):
                        raise RuntimeError(result)

        ray.get([actor.close.remote() for actor in actors])
    finally:
        ray.shutdown()
