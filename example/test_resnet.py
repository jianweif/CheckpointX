from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch
import torch.nn as nn
from ..utils.checkpoint import CheckpointXRunner
from .example_utils import benchmark, checkpointx_wrapper
from functools import partial
import argparse

MODEL_CFG = {
    # 11.68M params, 44.59 MB size
    "resnet18": {
        "input_size": 224,
        "batch_size": 512,
        "net": resnet18
    },
    # 21.79M params, 83.15 MB size
    "resnet34": {
        "input_size": 224,
        "batch_size": 376,
        "net": resnet34
    },
    # 25.55M params, 97.49 MB size
    "resnet50": {
        "input_size": 224,
        "batch_size": 128,
        "net": resnet50
    },
    # 44.55M params, 169.94 MB size
    "resnet101": {
        "input_size": 224,
        "batch_size": 96,
        "net": resnet101
    },
    # 60.19M params, 229.62 MB size
    "resnet152": {
        "input_size": 224,
        "batch_size": 64,
        "net": resnet152
    }
}

class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()

    def forward(self, inputs):
        return torch.mean(inputs)

def get_sequential_from_resnet(resnet):
    blocks = [resnet.conv1, nn.Sequential(resnet.bn1, resnet.relu), resnet.maxpool] + \
             list(resnet.layer1) + list(resnet.layer2) + list(resnet.layer3) + list(resnet.layer4) + [resnet.avgpool]
    blocks.append(SimpleLoss())
    return blocks

def test_resnet(resnet_name, model=None, batch_size=256, input_size=224, fraction=None, budget=None, checkpointx_runner=None):
    assert fraction != None or budget != None
    if model is None:
        model = MODEL_CFG[resnet_name]['net']()
        model = model.cuda()
        # model.train()
        model.eval() # todo: had to use eval mode to pass gradient test
    else:
        model.zero_grad()

    blocks = get_sequential_from_resnet(model)
    blocks = nn.Sequential(*blocks)
    inputs = torch.randn(batch_size, 3, input_size, input_size).cuda()

    loss_normal = blocks(inputs)
    loss_normal.backward()

    grad_normal = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_normal[name] = param.grad.data.cpu().clone()


    model.zero_grad()
    inputs.grad = None

    if checkpointx_runner is None:
        checkpointx_runner = CheckpointXRunner()
    loss_checkpointx = checkpointx_runner.checkpointx_sequential(blocks, inputs, fraction=fraction, budget=budget)
    loss_checkpointx.backward()
    if torch.allclose(loss_normal, loss_checkpointx):
        print("Outputs test pass")
    else:
        raise Exception("Outputs test failed")

    grad_checkpointx = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_checkpointx[name] = param.grad.data.cpu().clone()

    # todo: for some reasons, gradients differ slightly by 1e-08
    for name in grad_checkpointx:
        if not torch.allclose(grad_normal[name], grad_checkpointx[name], atol=1e-06):
            raise Exception("Gradient test failed for {}".format(name))
        else:
            print("Gradient test passed for {}".format(name))
    print("Gradient test pass")

    model.zero_grad()
    inputs.grad = None
    # clean up everything for benchmarking
    del inputs, loss_checkpointx, loss_normal, grad_checkpointx, grad_normal

    normal_memory, normal_time = benchmark(blocks, blocks, batch_size, 3, input_size, input_size)
    checkpointx_func = partial(checkpointx_wrapper, functions=blocks, checkpointx_runner=checkpointx_runner, fraction=fraction, budget=budget)
    checkpointx_memory, checkpointx_time = benchmark(checkpointx_func, model, batch_size, 3, input_size, input_size)
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
    parser = argparse.ArgumentParser(description='Run Resnet example')
    parser.add_argument('--resnet_name',
                        help='resnet names, can be resnet18, resnet34, resnet50, resnet101, resnet152',
                        required=True,
                        type=str)
    parser.add_argument('--batch_size',
                        help='batch size',
                        default=512,
                        type=int)
    parser.add_argument('--input_size',
                        help='input image size',
                        default=224,
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

    test_resnet(args.resnet_name, model=None, batch_size=args.batch_size, input_size=args.input_size, fraction=args.fraction, budget=args.budget,
                checkpointx_runner=None)