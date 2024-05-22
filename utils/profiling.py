import torch
import time
import numpy as np
from typing import Tuple, List, Union
import warnings
from copy import deepcopy

def get_tensor_size(tensor: torch.Tensor):
    """
    :param tensor: torch.Tensor
    :return: tensor size in bytes
    """
    return tensor.nelement() * tensor.element_size()

def get_dict_size(inputs: dict, ignore_unknown_type=True):
    """
    :param inputs: dict of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: size in bytes
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    size = 0
    for key, input in inputs.items():
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                size += get_tensor_size(input)
            elif isinstance(input, list) or isinstance(input, tuple):
                size += get_tuple_size(input, ignore_unknown_type=ignore_unknown_type)
            elif isinstance(input, dict):
                size += get_dict_size(input, ignore_unknown_type=ignore_unknown_type)
            else:
                raise NotImplementedError
        elif ignore_unknown_type:
            warnings.warn("type {} has been ignored in memory profiling".format(type(input)))
        else:
            raise NotImplementedError
    return size

def get_tuple_size(inputs: Union[Tuple, List], ignore_unknown_type=True):
    """
    :param inputs: tuple of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: size in bytes
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    size = 0
    for input in inputs:
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                size += get_tensor_size(input)
            elif isinstance(input, (list, tuple)):
                size += get_tuple_size(input, ignore_unknown_type=ignore_unknown_type)
            elif isinstance(input, dict):
                size += get_dict_size(input, ignore_unknown_type=ignore_unknown_type)
            else:
                raise NotImplementedError
        elif ignore_unknown_type:
            warnings.warn("type {} has been ignored in memory profiling".format(type(input)))
        else:
            raise NotImplementedError
    return size

def get_size(inputs: Union[Tuple, List, dict, torch.Tensor], ignore_unknown_type=True):
    """
    :param inputs: inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: size in bytes
    """
    if isinstance(inputs, torch.Tensor):
        return get_tensor_size(inputs)
    elif isinstance(inputs, (list, tuple)):
        return get_tuple_size(inputs, ignore_unknown_type=ignore_unknown_type)
    elif isinstance(inputs, dict):
        return get_dict_size(inputs, ignore_unknown_type=ignore_unknown_type)
    elif ignore_unknown_type:
        warnings.warn("type {} has been ignored in memory profiling".format(type(inputs)))
        return 0
    else:
        raise NotImplementedError

def make_tensor_grad(tensor: torch.Tensor):
    """
    :param tensor: torch.Tensor
    :return: tensor clone as gradients
    """
    return tensor.clone().detach()

def make_dict_grad(inputs: dict):
    """
    :param inputs: dict of inputs
    :return: dict of grads
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    grad_inputs = inputs
    for key, input in inputs.items():
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                grad_inputs[key] = make_tensor_grad(input)
            else:
                # ignore non tensors
                pass
    return grad_inputs

def make_tuple_grad(inputs: Union[Tuple, List]):
    """
    :param inputs: tuple of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: tuple/list of grads
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    grad_inputs = list(inputs)
    for i, input in enumerate(inputs):
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                grad_inputs[i] = make_tensor_grad(input)
            elif isinstance(input, (list, tuple)):
                grad_inputs[i] = make_tuple_grad(input)
            elif isinstance(input, dict):
                grad_inputs[i] = make_dict_grad(input)
    if isinstance(inputs, tuple):
        grad_inputs = tuple(grad_inputs)
    return grad_inputs

def make_grad(inputs: Union[Tuple, List, dict, torch.Tensor]):
    """
    :param inputs: inputs
    :return: grads
    """
    if isinstance(inputs, torch.Tensor):
        return make_tensor_grad(inputs)
    elif isinstance(inputs, (list, tuple)):
        return make_tuple_grad(inputs)
    elif isinstance(inputs, dict):
        return make_dict_grad(inputs)
    else:
        raise NotImplementedError

def remove_tensor_grad(tensor: torch.Tensor):
    """
    :param tensor: torch.Tensor
    :return:
    """
    tensor.grad = None

def remove_dict_grad(inputs: dict):
    """
    :param inputs: dict of inputs
    :return:
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    for key, input in inputs.items():
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                remove_tensor_grad(input)
            else:
                # ignore non tensors
                pass

def remove_tuple_grad(inputs: Union[Tuple, List]):
    """
    :param inputs: tuple of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return:
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    for i, input in enumerate(inputs):
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                remove_tensor_grad(input)
            elif isinstance(input, (list, tuple)):
                remove_tuple_grad(input)
            elif isinstance(input, dict):
                remove_dict_grad(input)

def remove_grad(inputs: Union[Tuple, List, dict, torch.Tensor]):
    """
    :param inputs: inputs
    :return:
    """
    if isinstance(inputs, torch.Tensor):
        return remove_tensor_grad(inputs)
    elif isinstance(inputs, (list, tuple)):
        return remove_tuple_grad(inputs)
    elif isinstance(inputs, dict):
        return remove_dict_grad(inputs)
    else:
        raise NotImplementedError

def detach_tensor(tensor: torch.Tensor):
    """
    :param tensor: torch.Tensor
    :return: detached tensor
    """
    return tensor.detach()

def detach_dict(inputs: dict):
    """
    :param inputs: dict of inputs
    :return: detached dict
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    for key, input in inputs.items():
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                inputs[key] = detach_tensor(input)
            else:
                # ignore non tensors
                pass
    return inputs

def detach_tuple(inputs: Union[Tuple, List]):
    """
    :param inputs: tuple of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: detach tuple/list
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    detach_inputs = list(inputs)
    for i, input in enumerate(inputs):
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                detach_inputs[i] = detach_tensor(input)
            elif isinstance(input, (list, tuple)):
                detach_inputs[i] = detach_tuple(input)
            elif isinstance(input, dict):
                detach_inputs[i] = detach_dict(input)
    if isinstance(inputs, tuple):
        detach_inputs = tuple(detach_inputs)
    return detach_inputs

def detach_inputs(inputs: Union[Tuple, List, dict, torch.Tensor]):
    """
    :param inputs: inputs
    :return: detached inputs
    """
    if isinstance(inputs, torch.Tensor):
        return detach_tensor(inputs)
    elif isinstance(inputs, (list, tuple)):
        return detach_tuple(inputs)
    elif isinstance(inputs, dict):
        return detach_dict(inputs)
    else:
        raise NotImplementedError

def requires_grad_tensor(tensor: torch.Tensor, requires_grad=True):
    """
    :param tensor: torch.Tensor
    :return: tensor with modified requires_grad
    """
    return tensor.requires_grad_(requires_grad)

def requires_grad_dict(inputs: dict, requires_grad=True):
    """
    :param inputs: dict of inputs
    :return: dict with modified requires_grad
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    for key, input in inputs.items():
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                inputs[key] = requires_grad_tensor(input, requires_grad=requires_grad)
            else:
                # ignore non tensors
                pass
    return inputs

def requires_grad_tuple(inputs: Union[Tuple, List], requires_grad=True):
    """
    :param inputs: tuple of inputs
    :param ignore_unknown_type: if True, skip the size of unknown types
    :return: tuple/list with modified requires_grad
    """
    supported_types = [torch.Tensor, list, tuple, dict]
    modified_inputs = list(inputs)
    for i, input in enumerate(inputs):
        if type(input) in supported_types:
            if isinstance(input, torch.Tensor):
                modified_inputs[i] = requires_grad_tensor(input, requires_grad=requires_grad)
            elif isinstance(input, (list, tuple)):
                modified_inputs[i] = requires_grad_tuple(input, requires_grad=requires_grad)
            elif isinstance(input, dict):
                modified_inputs[i] = requires_grad_dict(input, requires_grad=requires_grad)
    if isinstance(inputs, tuple):
        modified_inputs = tuple(modified_inputs)
    return modified_inputs

def requires_grad_inputs(inputs: Union[Tuple, List, dict, torch.Tensor], requires_grad=True):
    """
    :param inputs: inputs
    :return: inputs with modified requires_grad
    """
    if isinstance(inputs, torch.Tensor):
        return requires_grad_tensor(inputs, requires_grad=requires_grad)
    elif isinstance(inputs, (list, tuple)):
        return requires_grad_tuple(inputs, requires_grad=requires_grad)
    elif isinstance(inputs, dict):
        return requires_grad_dict(inputs, requires_grad=requires_grad)
    else:
        raise NotImplementedError

def benchmark_memory(function, device, *inputs, ignore_unknown_type=True, use_diff_for_output_mem=False):
    """
    :param function: runnable torch.nn.Module
    :param device: torch.device
    :param inputs: inputs to the function
    :param ignore_unknown_type: if True, skip the size of unknown types for memory profiling
    :return: outputs, forward memory for function, outputs memory (activation), backward peak memory for function, and formed
    gradient memory of function, all in bytes
    """
    if device.type != "cuda":
        # so far only support cuda memory profiling
        raise NotImplementedError

    torch.cuda.reset_peak_memory_stats(device)
    tmp_inputs = deepcopy(inputs)
    nograd_fwd_start_mem = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        outputs = function(*tmp_inputs)
    nograd_fwd_end_mem = torch.cuda.memory_allocated(device)
    nograd_fwd_max_mem = torch.cuda.max_memory_allocated(device)

    nograd_fwd_peak_mem = max(nograd_fwd_max_mem - nograd_fwd_start_mem, 0)

    if use_diff_for_output_mem:
        outputs_mem = nograd_fwd_end_mem - nograd_fwd_start_mem
    else:
        outputs_mem = get_size(outputs, ignore_unknown_type=ignore_unknown_type)
    # inputs_mem = get_size(inputs, ignore_unknown_type=ignore_unknown_type)

    del outputs, tmp_inputs


    torch.cuda.reset_peak_memory_stats(device)
    fwd_start_mem = torch.cuda.memory_allocated(device)
    inputs = requires_grad_inputs(inputs, requires_grad=True)
    with torch.enable_grad():
        outputs = function(*inputs)
    fwd_end_mem = torch.cuda.memory_allocated(device)
    fwd_max_mem = torch.cuda.max_memory_allocated(device)

    # denotes peak memory needed for forward, including outputs
    fwd_peak_mem = max(fwd_max_mem - fwd_start_mem, 0)
    # denotes stored activations for forward, excluding outputs
    fwd_act_mem = fwd_end_mem - fwd_start_mem - outputs_mem

    outputs_grad = make_grad(outputs)

    torch.cuda.reset_peak_memory_stats(device)
    bwd_start_mem = torch.cuda.memory_allocated(device)
    if isinstance(outputs, dict):
        for key in outputs:
            if outputs[key].requires_grad:
                torch.autograd.backward(outputs[key], outputs_grad[key], retain_graph=False)
    elif isinstance(outputs, torch.Tensor) or isinstance(outputs, tuple):
        torch.autograd.backward(outputs, outputs_grad, retain_graph=False)
    else:
        raise NotImplementedError
    bwd_end_mem = torch.cuda.memory_allocated(device)
    bwd_max_mem = torch.cuda.max_memory_allocated(device)
    bwd_peak_mem = bwd_max_mem - bwd_start_mem # refers to peak memory, and contain gradients for both function, and inputs

    # getting the right gradient memory for all trainable parameters in function
    grad_mem = 0
    for param in function.parameters():
        if param.grad is not None:
            grad_mem += param.grad.nelement() * param.grad.element_size()
    # gradient_mem = max(0, end_mem - start_mem - inputs_mem) # only count the gradient of function and excluding inputs and outputs

    function.zero_grad()
    remove_grad(inputs)
    return detach_inputs(outputs), nograd_fwd_peak_mem, fwd_act_mem, fwd_peak_mem, outputs_mem, bwd_peak_mem, grad_mem


def benchmark_time(function, device: torch.device, *inputs, warmup=3, repeat=5):
    """
    :param function: runnable torch.nn.Module
    :param device: torch.device
    :param inputs: inputs to the function
    :param warmup: warmup steps to run before benchmarking
    :param repeat: repeat steps for benchmarking
    :return: averaged runtime in seconds
    """
    assert repeat > 0
    if device.type not in ["cpu", "cuda"]:
        raise NotImplementedError
    with torch.no_grad():
        for _ in range(warmup):
            _ = function(*deepcopy(inputs))
        avg_time = []
        for _ in range(repeat):
            tmp_inputs = deepcopy(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()
            _ = function(*tmp_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()
            avg_time.append(end - start)
        return np.mean(np.array(avg_time))

def run_profiling(functions, device, *inputs, **kwargs):
    if device.type != "cuda":
        raise NotImplementedError

    grad_aware = kwargs.pop("grad_aware", True)

    fwd_ts = []
    nograd_fwd_peak_mems, fwd_act_mems, fwd_peak_mems, outputs_mems, bwd_peak_mems, grad_mems = [], [], [], [], [], []

    inputs = detach_inputs(inputs)
    for function in functions:
        fwd_t = benchmark_time(function, device, *deepcopy(inputs), **kwargs)
        outputs, nograd_fwd_peak_mem, fwd_act_mem, fwd_peak_mem, outputs_mem, bwd_peak_mem, grad_mem = benchmark_memory(function, device, *inputs, **kwargs)
        if isinstance(outputs, torch.Tensor):
            inputs = (outputs,)
        elif isinstance(outputs, dict):
            inputs = (outputs, )
        else:
            inputs = outputs
        fwd_ts.append(fwd_t)
        nograd_fwd_peak_mems.append(nograd_fwd_peak_mem)
        fwd_act_mems.append(fwd_act_mem)
        fwd_peak_mems.append(fwd_peak_mem)
        outputs_mems.append(outputs_mem)
        bwd_peak_mems.append(bwd_peak_mem)
        grad_mems.append(grad_mem)

    meta_data = {}
    meta_data["fwd_ts"] = np.array(fwd_ts)
    meta_data["nograd_fwd_peak_mems"] = np.array(nograd_fwd_peak_mems)
    meta_data["fwd_act_mems"] = np.array(fwd_act_mems)
    meta_data["fwd_peak_mems"] = np.array(fwd_peak_mems)
    meta_data["outputs_mems"] = np.array(outputs_mems)
    meta_data["bwd_peak_mems"] = np.array(bwd_peak_mems)
    meta_data["grad_mems"] = np.array(grad_mems)
    if not grad_aware:
        meta_data["grad_mems"][:] = 0.0

    return meta_data