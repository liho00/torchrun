# torchrun_flux.py
import time
import torch
from common_dtensor import DTensor
import torch.distributed as dist
from diffusers import FluxPipeline

class DiffusionPipelineTest(DTensor):
    @property
    def world_size(self):
        device_count = torch.cuda.device_count()
        if device_count <= 4:
            return device_count
        elif device_count < 6:
            return 4
        elif device_count < 8:
            return 6
        else:
            return 8

    def mesh(self, device, use_batch, use_ring):
        from para_attn.context_parallel import init_context_parallel_mesh

        max_batch_dim_size = None
        if use_batch:
            max_batch_dim_size = 2
        max_ring_dim_size = None
        if use_ring:
            max_ring_dim_size = 2
        mesh = init_context_parallel_mesh(
            device, max_batch_dim_size=max_batch_dim_size, max_ring_dim_size=max_ring_dim_size
        )
        return mesh

    def new_pipe(self, dtype, device, rank):
        raise NotImplementedError

    def call_pipe(self, pipe, *args, **kwargs):
        raise NotImplementedError

    def _benchmark_pipe(self, dtype, device, parallelize, compile, use_batch, use_ring):
        torch.manual_seed(0)

        pipe = self.new_pipe()

        if parallelize:
            from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

            mesh = self.mesh(device, use_batch=use_batch, use_ring=use_ring)
            parallelize_pipe(pipe, mesh=mesh)

        if compile:
            if parallelize:
                torch._inductor.config.reorder_for_compute_comm_overlap = True
            # If cudagraphs is enabled and parallelize is True, the test will hang indefinitely
            # after the last iteration.
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

        for _ in range(2):
            begin = time.time()
            images = self.call_pipe(pipe)
            end = time.time()
            if self.rank == 0:
                images[0].save("cat.jpg")
                print(f"Time taken: {end - begin:.3f} seconds")

class FluxPipelineTest(DiffusionPipelineTest):
    def new_pipe(self):
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        ).to(f"cuda:{self.rank}")
        return pipe

    def call_pipe(self, pipe, *args, **kwargs):
        return pipe(
            "A cat holding a sign that says hello world",
            num_inference_steps=8,
            output_type="pil" if self.rank == 0 else "latent",
        ).images

    def benchmark_pipe(self):
        DEVICE_TYPE = (
            "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
        )
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.world_size:
            self.device_type = "cpu"
        else:
            self.device_type = DEVICE_TYPE

        self.init_pg()
        try:
            super()._benchmark_pipe(torch.bfloat16, "cuda", True, False, False, True)
        except Exception as e:
            dist.destroy_process_group()
            raise e

        self.destroy_pg()


if __name__ == "__main__":
    flux = FluxPipelineTest()
    flux._start_processes("benchmark_pipe")
