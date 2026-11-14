import torch
import gc
import time
import threading
import statistics
from torch import nn
import calflops


def profile(model: nn.Module, input_shape, device: torch.device):
    with torch.no_grad():
        model.to(device)
        model.eval()
        cuda = device.type == "cuda"
        warmup_its = 10
        bench_its = 20

        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

        x = torch.rand(*input_shape, device=device)

        for _ in range(warmup_its):
            model(x)

        if cuda:
            torch.cuda.synchronize()

        flops, macs, _ = calflops.calculate_flops(
            model=model, input_shape=input_shape, print_results=False, output_as_string=False
        )
        params = sum(p.numel() for p in model.parameters()) 

        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

        latencies = [] # in seconds
        peak_mem_usages = [] #in bytes
        mem_usages = [] # in bytes
        power_samples = []  #in milliwatts

        for _ in range(bench_its):
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            stop_flag = threading.Event()
            run_power_samples = []
            run_mem_usages = []
            start_mem = torch.cuda.memory_allocated() if cuda else 0

            def sample_power():
                while not stop_flag.is_set():
                    try:
                        run_power_samples.append(torch.cuda.power_draw())
                        run_mem_usages.append(torch.cuda.memory_allocated() - start_mem)
                    except RuntimeError:
                        pass
                    stop_flag.wait(0.01)

            if cuda:
                sampler = threading.Thread(target=sample_power)
                sampler.start()

            start_time = time.perf_counter()
            model(x)

            if cuda:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            if cuda:
                peak_mem_usages.append(torch.cuda.max_memory_allocated() - start_mem)
                stop_flag.set()
                sampler.join()

            power_samples.extend(run_power_samples)
            mem_usages.extend(run_mem_usages)

        total_latency = statistics.median(latencies)
        peak_mem = statistics.median(peak_mem_usages) if peak_mem_usages else 0

        mem_avg = sum(mem_usages) / len(mem_usages) if mem_usages else 0.0
        avg_power_w = sum(power_samples) / len(power_samples) / 1000 if power_samples else 0.0
        energy_j = avg_power_w * total_latency

        return {
            "name": model.__class__.__name__,
            "total_latency": total_latency,
            "parameters": params,
            "macs": macs,
            "flops": flops,
            "gpu_mem_peak": peak_mem,
            "gpu_mem_avg": mem_avg,
            "avg_power_w": avg_power_w,
            "energy_j": energy_j
        }
