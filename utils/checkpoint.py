import numpy as np
import torch
from .profiling import run_profiling, requires_grad_inputs, detach_inputs
from .solver import preprocess_meta_data, convert_memory_in_metadata, compute_budget_lower_bound, solver_bottom_up_vectorized, \
    parse_solution_seq_exec, RunType, BackwardType, compute_regular_forward_lower_bound
from torch.utils.checkpoint import check_backward_validity, _infer_device_type, _get_autocast_kwargs, _get_device_module, \
    get_device_states, set_device_states, _supports_autocast
import warnings
import contextlib

@torch.no_grad()
def nograd_forward(function_list, *args):
    x = args
    for i, layer in enumerate(function_list):
        if isinstance(x, torch.Tensor) or isinstance(x, dict):
            x = (x, )
        x = layer(*x)
        # todo: handle tuple or list here
    return x

@torch.enable_grad()
def regular_forward(function_list, *args):
    if len(function_list) == 0:
        raise Exception("cannot forward 0 layers")
    x = args
    for i, layer in enumerate(function_list):
        if isinstance(x, torch.Tensor) or isinstance(x, dict):
            x = (x,)
        x = layer(*x)
    return x

def regular_backward(functions, detached_inputs, *grad_outputs):
    detached_inputs = requires_grad_inputs(detached_inputs, requires_grad=True)
    if isinstance(detached_inputs, torch.Tensor) or isinstance(detached_inputs, dict):
        detached_inputs = (detached_inputs,)

    outputs = regular_forward(functions, *detached_inputs)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)

    outputs_with_grad = []
    grad_outputs_with_grad = []
    for i in range(len(outputs)):
        if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
            outputs_with_grad.append(outputs[i])
            grad_outputs_with_grad.append(grad_outputs[i])
    if len(outputs_with_grad) == 0:
        raise RuntimeError(
            "none of output has requires_grad=True,"
            " this checkpoint() is not necessary"
        )
    torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad, retain_graph=False)
    grad_outputs = []
    for inp in detached_inputs:
        if hasattr(inp, "grad"):
            grad_outputs.append(inp.grad)
        else:
            grad_outputs.append(None)
    grad_outputs = tuple(grad_outputs)

    del outputs_with_grad, outputs, grad_outputs_with_grad
    del detached_inputs

    return grad_outputs

class CheckpointXFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, functions, start, end, backward_exec_seq, preserve_rng_state, *inputs):
        check_backward_validity(inputs)
        ctx.functions = functions
        ctx.backward_exec_seq = backward_exec_seq
        ctx.start = start
        ctx.end = end
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*inputs)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # functions, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*inputs)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(inputs):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        return detach_inputs(nograd_forward(functions[start:end], *inputs))

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument."
            )
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_inputs(tuple(inputs))

            device_autocast_ctx = device_module.amp.autocast(
                **ctx.device_autocast_kwargs
            ) if _supports_autocast(ctx.device) else contextlib.nullcontext()

            # todo: modify below


            with device_autocast_ctx, \
                 torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                inputs_cache = {ctx.start: detached_inputs}
                inputs_loc = [ctx.start]
                for backward_type, seg_start, seg_end in ctx.backward_exec_seq:
                    if backward_type == BackwardType.Normal:
                        grad_outputs = regular_backward(ctx.functions[seg_start:seg_end], inputs_cache[seg_start], *grad_outputs)
                        del inputs_cache[seg_start] # free up tensor
                        if inputs_loc[-1] == seg_start:
                            inputs_loc.pop()
                    elif backward_type == BackwardType.Checkpoint:
                        if seg_end not in inputs_cache:
                            closest_inputs_loc = inputs_loc[-1]
                            inputs_seg_end = nograd_forward(ctx.functions[closest_inputs_loc:seg_end], *inputs_cache[closest_inputs_loc])
                            if isinstance(inputs_seg_end, torch.Tensor):
                                # change to tuple
                                inputs_seg_end = (inputs_seg_end,)
                            inputs_cache[seg_end] = inputs_seg_end
                            del inputs_seg_end
                            inputs_loc.append(seg_end)
                    else:
                        raise NotImplementedError
        return (None, None, None, None, None) + grad_outputs

class CheckpointXRunner:
    def __init__(self, **kwargs):
        # todo: add different initialization method
        self.profiled = False
        self.meta_data = None
        self.solved = False
        self.budget = None # todo: do better saving here
        self.solution = None
        self.dp = None
        self.parsed_solution = None

    def summarize(self, budget=None):
        if budget is None:
            budget = self.budget
        full_mem = compute_regular_forward_lower_bound(self.meta_data, 0, len(self.meta_data["fwd_ts"]))
        regular_time = sum(self.meta_data["fwd_ts"])
        optimal_time = self.dp[budget, 0, -1]
        print("[Theoretical] Full memory {} MB, budget {} MB, reduced to {:.2f}%, Regular forward time {:.6f}, optimal additional forward time {:.6f}, "
              "increased to {:.2f}%".format(full_mem, budget, budget * 100 / full_mem, regular_time, optimal_time, (optimal_time / regular_time + 1) * 100))
        return {
            "theo_reg_mem": full_mem,
            "theo_ckx_mem": budget,
            "theo_reg_time": regular_time,
            "theo_ckx_time": optimal_time + regular_time,
        }

    def checkpointx_sequential(self, functions, *inputs, preserve_rng_state=True, **kwargs):
        preserve = preserve_rng_state
        budget = kwargs.pop("budget", None)
        fraction = kwargs.pop("fraction", None)
        grad_aware = kwargs.pop("grad_aware", True)
        assert budget is not None or fraction is not None

        device = inputs[0].device # todo: better way to get
        if self.parsed_solution is None:
            if not self.profiled:
                meta_data = run_profiling(functions, device, *inputs, grad_aware=grad_aware)
                meta_data = convert_memory_in_metadata(meta_data,
                                                       scale=1024 ** 2)  # todo: temporary hard coded here, open to input later
                meta_data = preprocess_meta_data(meta_data)
                self.meta_data = meta_data
                self.profiled = True

            if budget is None:
                if fraction is None:
                    raise Exception("fraction or budget needs to be provided")
                full_mem = compute_regular_forward_lower_bound(self.meta_data, 0, len(self.meta_data["fwd_ts"]))
                budget_lower_bound = compute_budget_lower_bound(self.meta_data)
                budget_lower_bound = min(budget_lower_bound, full_mem)
                if budget_lower_bound > full_mem:
                    raise Exception("Detected full_mem < budget_lower_bound")
                budget = int(full_mem * fraction)
                if budget < budget_lower_bound:
                    warnings.warn("Provided budget {} is lower than estimated budget lower bound {}".format(budget,
                                                                                                            budget_lower_bound))
                    warnings.warn("Setting budget to budget lower bound {}".format(budget_lower_bound))
                    budget = budget_lower_bound
            dp, solution = solver_bottom_up_vectorized(self.meta_data, budget)
            if dp[-1, 0, -1] == np.inf:
                raise Exception("no valid solution at budget {}".format(budget))
            parsed_solution = parse_solution_seq_exec(dp, solution)
            self.solved = True
            self.budget = budget
            self.dp = dp
            self.solution = solution
            self.parsed_solution = parsed_solution
        else:
            if budget is None:
                if fraction is None:
                    raise Exception("fraction or budget needs to be provided")
                full_mem = compute_regular_forward_lower_bound(self.meta_data, 0, len(self.meta_data["fwd_ts"]))
                # budget_lower_bound = get_budget_lower_bound(self.meta_data)
                # budget_lower_bound = compute_nograd_forward_lower_bound(self.meta_data, 0, len(self.meta_data["fwd_ts"]))
                budget_lower_bound = compute_budget_lower_bound(self.meta_data)
                budget_lower_bound = min(full_mem, budget_lower_bound)
                if budget_lower_bound > full_mem:
                    raise Exception("Detected full_mem < budget_lower_bound")

                budget = int(full_mem * fraction)
                if budget < budget_lower_bound:
                    warnings.warn("Provided budget {} is lower than estimated budget lower bound {}".format(budget,
                                                                                                            budget_lower_bound))
                    warnings.warn("Setting budget to budget lower bound {}".format(budget_lower_bound))
                    budget = budget_lower_bound
            if budget > self.budget:
                dp, solution = solver_bottom_up_vectorized(self.meta_data, budget)
                parsed_solution = parse_solution_seq_exec(dp, solution)
                self.solved = True
                self.budget = budget
                self.dp = dp
                self.solution = solution
                self.parsed_solution = parsed_solution
            else:
                # can be parsed from existing solution
                parsed_solution = parse_solution_seq_exec(self.dp[:(budget + 1)], self.solution[:(budget + 1)])
                self.parsed_solution = parsed_solution

        x = inputs
        for run_info in self.parsed_solution:
            if run_info[0] == RunType.Normal:
                start, end = run_info[1], run_info[2]
                if isinstance(x, torch.Tensor) or isinstance(x, dict):
                    x = (x,)
                x = requires_grad_inputs(x)
                x = regular_forward(functions[start:end], *x)
            else:
                start, end, backward_exec_seq = run_info[1], run_info[2], run_info[3]
                if isinstance(x, torch.Tensor) or isinstance(x, dict):
                    x = (x,)
                x = requires_grad_inputs(x)
                x = CheckpointXFunction.apply(functions, start, end, backward_exec_seq, preserve, *x)
        return x