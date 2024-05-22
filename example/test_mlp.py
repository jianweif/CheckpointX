import torch
from ..utils.checkpoint import CheckpointXRunner
from .model.mlp import MLP
from .example_utils import benchmark, checkpointx_wrapper
from functools import partial
import argparse

def test_deep_mlp(n_layer=32, mlp=None, batch_size=2**20, dim=2**5, fraction=None, budget=None, checkpointx_runner=None, grad_aware=False):
    assert fraction != None or budget != None
    # n_layer = 64
    # batch_size = 2 ** 20
    # dim = 2 ** 5
    # fraction = (64) / 64
    if mlp is None:
        mlp = MLP(n_layer, dim)
        mlp = mlp.cuda()
        mlp.train()
    else:
        mlp.zero_grad()

    inputs = torch.randn(batch_size, dim).cuda()

    loss_normal = mlp(inputs)
    loss_normal.backward()

    grad_normal = {}
    for name, param in mlp.named_parameters():
        grad_normal[name] = param.grad.data.clone().cpu()

    mlp.zero_grad()
    inputs.grad = None
    torch.cuda.reset_peak_memory_stats()

    if checkpointx_runner is None:
        checkpointx_runner = CheckpointXRunner()
    loss_checkpointx = checkpointx_runner.checkpointx_sequential(mlp.net, inputs, fraction=fraction, budget=budget, grad_aware=grad_aware)
    loss_checkpointx.backward()

    if torch.allclose(loss_normal, loss_checkpointx):
        print("Outputs test pass")
    else:
        raise Exception("Outputs test failed")

    grad_checkpointx = {}
    for name, param in mlp.named_parameters():
        grad_checkpointx[name] = param.grad.data.clone().cpu()

    for name in grad_checkpointx:
        if not torch.allclose(grad_normal[name], grad_checkpointx[name]):
            raise Exception("Gradient test failed")
    print("Gradient test pass")

    mlp.zero_grad()
    inputs.grad = None
    # clean up everything for benchmarking
    del inputs, loss_checkpointx, loss_normal, grad_checkpointx, grad_normal

    normal_memory, normal_time = benchmark(mlp, mlp, batch_size, dim)
    checkpointx_func = partial(checkpointx_wrapper, functions=mlp.net, checkpointx_runner=checkpointx_runner, fraction=fraction, budget=budget)
    checkpointx_memory, checkpointx_time = benchmark(checkpointx_func, mlp, batch_size, dim)
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
    parser = argparse.ArgumentParser(description='Run MLP example')
    parser.add_argument('--n_layer',
                        help='number of layers for MLP',
                        required=True,
                        type=str)
    parser.add_argument('--batch_size',
                        help='batch size',
                        default=4096,
                        type=int)
    parser.add_argument('--dim',
                        help='dimension for the MLP',
                        default=2048,
                        type=int)
    parser.add_argument('--fraction',
                        help='fraction of memory after reduction, for example, 0.7 means reducing the memory to 70%',
                        default=0.5,
                        type=float)
    parser.add_argument('--budget',
                        help='memory budget in MB, will be used if specified and fraction will be ignored',
                        default=None,
                        type=int)

    args = parser.parse_args()
    test_deep_mlp(n_layer=args.n_layer, mlp=None, batch_size=args.batch_size, dim=args.dim, fraction=args.fraction, budget=args.budget, grad_aware=True)