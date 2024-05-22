import torch
from torch.utils.checkpoint import checkpoint_sequential
import time
import numpy as np
from tqdm import tqdm

def benchmark_single_iter(run_function, net, *input_dim):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    inputs = torch.randn(*input_dim).cuda()
    inputs.requires_grad_(True)

    start_memory = torch.cuda.max_memory_allocated()

    torch.cuda.synchronize()
    start_time = time.time()

    loss = run_function(inputs)
    if loss.nelement() > 1:
        # default to simple loss
        loss = torch.mean(loss)
    loss.backward()

    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.max_memory_allocated()

    if inputs.grad is not None:
        end_memory -= inputs.grad.nelement() * inputs.grad.element_size()

    net.zero_grad()
    inputs.grad = None

    # only contains activation memory and gradients, not having parameter size itself
    return (end_memory - start_memory) / (1024**2), end_time - start_time

def benchmark(run_function, net, *input_dim, warmup=5, repeat=20):
    for i in tqdm(range(warmup)):
        benchmark_single_iter(run_function, net, *input_dim)
    run_memory, run_time = [], []
    for i in tqdm(range(repeat)):
        m, t = benchmark_single_iter(run_function, net, *input_dim)
        run_memory.append(m)
        run_time.append(t)

    return np.max(run_memory), np.mean(run_time)

def checkpointx_wrapper(inputs, functions, checkpointx_runner, **kwargs):
    return checkpointx_runner.checkpointx_sequential(functions, inputs, **kwargs)

def checkpoint_sequential_wrapper(inputs, functions, segments):
    return checkpoint_sequential(functions, segments, inputs)