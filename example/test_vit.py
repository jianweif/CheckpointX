import torch
from ..utils.checkpoint import CheckpointXRunner
from .model.transformer_custom import CustomVisionTransformer
from .example_utils import benchmark, checkpointx_wrapper
from functools import partial
import time
from copy import deepcopy
import numpy as np
import argparse

MODEL_CFG = {
    # 86.1M params, 328 MB size
    "ViT-B-16": {
        "image_size": 224,
        "layers": 12,
        "width": 768,
        "heads": 768 // 64,
        "mlp_ratio": 4.0,
        "patch_size": 16,
        "batch_size": 96
    },
    # 304M params, 1158 MB size
    "ViT-L-14": {
        "image_size": 224,
        "layers": 24,
        "width": 1024,
        "heads": 1024 // 64,
        "mlp_ratio": 4.0,
        "patch_size": 14,
        "batch_size": 28
    },
    # 631M params, 2408 MB size
    "ViT-H-14": {
        "image_size": 224,
        "layers": 32,
        "width": 1280,
        "heads": 1280 // 80,
        "mlp_ratio": 4.0,
        "patch_size": 14,
        "batch_size": 12
    },
    # 1.01B params, 3860 MB size
    "ViT-g-14": {
        "image_size": 224,
        "layers": 40,
        "width": 1408,
        "heads": 1408 // 88,
        "mlp_ratio": 4.3637,
        "patch_size": 14,
        "batch_size": 8
    },
}

def test_deep_vit(vit_name, model=None, fraction=None, budget=None, checkpointx_runner=None):
    assert fraction != None or budget != None
    # n_layer = 64
    # batch_size = 2 ** 20
    # dim = 2 ** 5
    # fraction = (64) / 64
    cfg = deepcopy(MODEL_CFG[vit_name])
    batch_size = cfg.pop("batch_size")
    image_size = cfg["image_size"]
    if model is None:
        model = CustomVisionTransformer(**cfg)
        model = model.cuda()
        model.train()
    else:
        model.zero_grad()

    inputs = torch.randn(batch_size, 3, image_size, image_size).cuda()

    loss_normal = model(inputs)
    loss_normal.backward()

    grad_normal = {}
    for name, param in model.named_parameters():
        grad_normal[name] = param.grad.data.clone().cpu()

    # todo: debug only
    # print("Projected mem {} MB".format(torch.cuda.memory_allocated() / 1024 ** 2 + budget))

    model.zero_grad()
    inputs.grad = None

    if checkpointx_runner is None:
        checkpointx_runner = CheckpointXRunner()

    with torch.no_grad():
        seq = model.get_sequential_v3(inputs)
    loss_checkpointx = checkpointx_runner.checkpointx_sequential(seq, inputs, fraction=fraction, budget=budget)
    loss_checkpointx.backward()
    if torch.allclose(loss_normal, loss_checkpointx):
        print("Outputs test pass")
    else:
        raise Exception("Outputs test failed")

    grad_checkpointx = {}
    for name, param in model.named_parameters():
        grad_checkpointx[name] = param.grad.data.clone().cpu()

    for name in grad_checkpointx:
        if not torch.allclose(grad_normal[name], grad_checkpointx[name], atol=1e-05, rtol=1e-03):
            raise Exception("Gradient test failed at {}, max_diff {}".format(name, torch.max(torch.abs(grad_normal[name] - grad_checkpointx[name])).item()))
    print("Gradient test pass")

    model.zero_grad()
    inputs.grad = None
    # clean up everything for benchmarking
    del inputs, loss_checkpointx, loss_normal, grad_checkpointx, grad_normal

    normal_memory, normal_time = benchmark(model, model, batch_size, 3, image_size, image_size)
    checkpointx_func = partial(checkpointx_wrapper, functions=seq, checkpointx_runner=checkpointx_runner, fraction=fraction, budget=budget)
    checkpointx_memory, checkpointx_time = benchmark(checkpointx_func, model, batch_size, 3, image_size, image_size)
    res = checkpointx_runner.summarize(budget=budget)
    print(
        "[Measured] Full memory {} MB, checkpointx {} MB, reduced to {:.2f}%, Regular forward time {:.6f}, checkpointx forward time {:.6f}, "
        "increased to {:.2f}%".format(normal_memory, checkpointx_memory, checkpointx_memory * 100 / normal_memory, normal_time, checkpointx_time,
                                      (checkpointx_time / normal_time) * 100))
    res["measured_reg_mem"] = normal_memory
    res["measured_ckx_mem"] = checkpointx_memory
    res["measured_reg_time"] = normal_time
    res["measured_ckx_time"] = checkpointx_time
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ViT example')
    parser.add_argument('--vit_name',
                        help='vit names, can be ViT-B-16, ViT-L-14, ViT-H-14, ViT-g-14',
                        required=True,
                        type=str)
    parser.add_argument('--fraction',
                        help='fraction of memory after reduction, for example, 0.7 means reducing the memory to 70%',
                        default=0.5,
                        type=float)
    parser.add_argument('--budget',
                        help='memory budget in MB, will be used if specified and fraction will be ignored',
                        default=None,
                        type=int)

    args = parser.parse_args()

    test_deep_vit(args.vit_name, fraction=args.fraction, budget=args.budget)
