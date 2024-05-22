import numpy as np
from enum import IntEnum
from anytree import Node
from tqdm import tqdm

class ForwardType(IntEnum):
    Regular = 1
    Checkpoint = 2
    Infeasible = 3
    Recursive = 4
    Mixed = 5

class RunType(IntEnum):
    Normal = 1
    Checkpointx = 2

class BackwardType(IntEnum):
    Normal = 1
    Checkpoint = 2

def convert_memory(mem, scale=1024**2):
    if isinstance(mem, np.ndarray):
        return (mem / scale).astype(int)
    else:
        return int(mem / scale)

def convert_memory_in_metadata(meta_data, scale=1024**2):
    for key in meta_data:
        if "mem" in key:
            meta_data[key] = convert_memory(meta_data[key], scale=scale)

    return meta_data

def get_sum(cumsum, i, j):
    """
    :param cumsum:
    :param i:
    :param j:
    :return: return sum of [i, j)
    """
    n = len(cumsum)
    if not (i < j and j <= n):
        if i == j:
            return 0
        else:
            raise ValueError
    if i == 0:
        return cumsum[j - 1]
    else:
        return cumsum[j - 1] - cumsum[i - 1]


def create_max_dict_of_backward(backward_mems, gradient_mem_cumsum):
    # denote minimum memory needed for completing backward of segment [i, j)
    n = len(backward_mems)
    backward_mems_max_dict = {}
    for j in range(n, 0, -1):
        prev_max = 0
        for i in range(j - 1, -1, -1):
            # for segment [i, j)
            # computing max_k\in [i, j) b_k + \sum_t={k+1}^{j-1}g_t
            gradient_cumsum = 0 if i + 1 > j - 1 else get_sum(gradient_mem_cumsum, i + 1, j)
            cur_max = max(prev_max, backward_mems[i] + gradient_cumsum)
            backward_mems_max_dict[(i, j)] = cur_max
            prev_max = cur_max
    return backward_mems_max_dict

def compute_reg_fwd_low_bound_dict(meta_data):
    """
    Compute regular forward memory lower bound for arbitrary segments [i, j)
    :param meta_data:
    :return: dict of regular forward lower bound for arbitrary segments [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    regular_forward_lower_bound_dict = {}
    for j in range(n, 0, -1):
        for i in range(j - 1, -1, -1):
            # simulates forward and backward for segment [i, j)
            # simulates forward
            fwd_peak_mem_ij = 0
            for k in range(i, j):
                # forwarding kth layer
                fwd_act_mem_cumsum_ik = get_sum(meta_data["fwd_act_mems_cumsum"], i, k) + \
                                        get_sum(meta_data["outputs_mems_cumsum"], i, k)
                fwd_peak_mem_k = meta_data["fwd_peak_mems"][k]
                fwd_peak_mem_ij = max(fwd_peak_mem_ij, fwd_act_mem_cumsum_ik + fwd_peak_mem_k)

            bwd_peak_mem_ij = 0
            for k in range(j - 1, i - 1, -1):
                # backwarding kth layer
                fwd_act_mem_cumsum_ik = get_sum(meta_data["fwd_act_mems_cumsum"], i, k) + \
                                        get_sum(meta_data["outputs_mems_cumsum"], i, k)
                # accumulated gradients
                grad_mem_cumsum_kj = get_sum(meta_data["grad_mems_cumsum"], k + 1, j)
                bwd_peak_mem_k = meta_data["bwd_peak_mems"][k]
                bwd_peak_mem_ij = max(bwd_peak_mem_ij, fwd_act_mem_cumsum_ik + bwd_peak_mem_k + grad_mem_cumsum_kj)

            # final state
            bwd_peak_mem_ij = max(bwd_peak_mem_ij, get_sum(meta_data["grad_mems_cumsum"], i, j))
            regular_forward_lower_bound_dict[(i, j)] = max(fwd_peak_mem_ij, bwd_peak_mem_ij)

    return regular_forward_lower_bound_dict

def compute_nograd_fwd_low_bound_dict(meta_data):
    """
    Compute no gradients forward memory lower bound for arbitrary segments [i, j)
    :param meta_data:
    :return: dict of no gradients forward lower bound for arbitrary segments [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    nograd_forward_lower_bound_dict = {}
    for j in range(n, 0, -1):
        for i in range(j - 1, -1, -1):
            # simulates no grad forward and backward for segment [i, j)
            # simulate peak memory for 1. nograd forward for [i, k); 2. regular forward for k; 3. regular backward for k
            peak_mem_ij = 0
            for k in range(j - 1, i - 1, -1):
                if k > i:
                    if i - 1 >= 0:
                        nograd_fwd_peak_mem_ik = np.max(meta_data["outputs_mems"][(i-1):(k-1)] + meta_data["nograd_fwd_peak_mems"][i:k])
                    else:
                        if k - 1 > i:
                            nograd_fwd_peak_mem_ik = np.max(meta_data["outputs_mems"][i:(k-1)] + meta_data["nograd_fwd_peak_mems"][(i+1):k])
                            nograd_fwd_peak_mem_ik = max(nograd_fwd_peak_mem_ik, meta_data["nograd_fwd_peak_mems"][i])
                        else:
                            nograd_fwd_peak_mem_ik = meta_data["nograd_fwd_peak_mems"][i]
                else:
                    nograd_fwd_peak_mem_ik = 0

                # memory for forwarding and backwarding k regularly, not containing inputs memory
                reg_fwd_low_bound_k = meta_data["reg_fwd_low_bound"][(k, k + 1)]
                # get the input memory for k but exclude the very first inputs at i
                inputs_mem_k = meta_data["outputs_mems"][k - 1] if k > i else 0

                # accumulated gradients
                grad_mem_cumsum_kj = get_sum(meta_data["grad_mems_cumsum"], k + 1, j)

                peak_mem_ij = max(peak_mem_ij, grad_mem_cumsum_kj +
                                  max(nograd_fwd_peak_mem_ik, inputs_mem_k + reg_fwd_low_bound_k))
            # final state
            peak_mem_ij = max(peak_mem_ij, get_sum(meta_data["grad_mems_cumsum"], i, j))
            nograd_forward_lower_bound_dict[(i, j)] = peak_mem_ij

    return nograd_forward_lower_bound_dict

def compute_left_budget_offset_option1(meta_data):
    """
    Computes left budget offset for option1 for faster solver
    :param meta_data:
    :return: left_budget_offset_option1 is matrix of shape [n + 1, n + 1]
    element i, j denotes corresponding left budget offset for option1 for segment [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    left_budget_offset_option1 = -np.ones([n + 1, n + 1], dtype=meta_data["grad_mems_cumsum"].dtype)
    for i in range(n):
        for j in range(i + 1, n + 1):
            left_budget_offset_option1[i, j] = get_sum(meta_data["grad_mems_cumsum"], i, j) + meta_data["outputs_mems"][i]
    return left_budget_offset_option1

def compute_left_budget_offset_option2(meta_data):
    """
    Computes left budget offset for option2 for faster solver
    :param meta_data:
    :return: left_budget_offset_option2 is matrix of shape [n + 1, n + 1]
    element i, j denotes corresponding left budget offset for option2 for segment [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    left_budget_offset_option2 = -np.ones([n + 1, n + 1], dtype=meta_data["grad_mems_cumsum"].dtype)
    for i in range(n):
        for j in range(i + 1, n + 1):
            left_budget_offset_option2[i, j] = get_sum(meta_data["grad_mems_cumsum"], i, j)
    return left_budget_offset_option2

def compute_right_budget_offset_option2(meta_data):
    """
    Computes left budget offset for option2 for faster solver
    :param meta_data:
    :return: left_budget_offset_option2 is matrix of shape [n + 1, n + 1]
    element i, j denotes corresponding left budget offset for option2 for segment [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    right_budget_offset_option2 = -np.ones([n + 1, n + 1], dtype=meta_data["fwd_act_mems_cumsum"].dtype)
    for i in range(n):
        for j in range(i + 1, n + 1):
            right_budget_offset_option2[i, j] = get_sum(meta_data["fwd_act_mems_cumsum"], i, j) + get_sum(meta_data["outputs_mems_cumsum"], i, j)
    return right_budget_offset_option2


def compute_reg_fwd_lo_bound_mat(meta_data):
    """
    Computes regular forward lower bound for option2 for faster solver
    :param meta_data:
    :return: reg_fwd_lo_bound_mat is matrix of shape [n + 1, n + 1]
    element i, j denotes corresponding regular forward lower bound for segment [i, j)
    """
    n = len(meta_data["bwd_peak_mems"])
    reg_fwd_lo_bound_mat = -np.ones([n + 1, n + 1], dtype=meta_data["fwd_act_mems_cumsum"].dtype)
    for i in range(n):
        for j in range(i + 1, n + 1):
            reg_fwd_lo_bound_mat[i, j] = compute_regular_forward_lower_bound(meta_data, i, j)
    return reg_fwd_lo_bound_mat

def compute_remat_time_option1(meta_data):
    """
    Computes remat time for option1 for faster solver
    :param meta_data:
    :return: left_budget_offset_option1 is matrix of shape [n + 1, n + 1]
    element i, j denotes corresponding remat time for option1 for segment [i, j), to rematerialize tensor j from tensor i
    """
    n = len(meta_data["bwd_peak_mems"])
    remat_time_option1 = -np.ones([n + 1, n + 1], dtype=meta_data["fwd_ts_cumsum"].dtype)
    for i in range(n):
        for j in range(i + 1, n + 1):
            remat_time_option1[i, j] = get_sum(meta_data["fwd_ts_cumsum"], i, j)
    return remat_time_option1

def preprocess_meta_data(meta_data):
    """
    :return: meta_data: a dict of meta data
    """

    meta_data["fwd_ts_cumsum"] = np.cumsum(meta_data["fwd_ts"])
    meta_data["fwd_act_mems_cumsum"] = np.cumsum(meta_data["fwd_act_mems"])
    meta_data["outputs_mems_cumsum"] = np.cumsum(meta_data["outputs_mems"])
    meta_data["grad_mems_cumsum"] = np.cumsum(meta_data["grad_mems"])

    meta_data["reg_fwd_low_bound"] = compute_reg_fwd_low_bound_dict(meta_data)
    meta_data["nograd_fwd_low_bound"] = compute_nograd_fwd_low_bound_dict(meta_data)
    meta_data["left_budget_offset_option1"] = compute_left_budget_offset_option1(meta_data)
    meta_data["remat_time_option1"] = compute_remat_time_option1(meta_data)
    meta_data["left_budget_offset_option2"] = compute_left_budget_offset_option2(meta_data)
    meta_data["reg_fwd_lo_bound_mat"] = compute_reg_fwd_lo_bound_mat(meta_data)
    meta_data["right_budget_offset_option2"] = compute_right_budget_offset_option2(meta_data)

    return meta_data

def compute_regular_forward_lower_bound(meta_data, start, end):
    """
    :param meta_data:
    :param start:
    :param end:
    :return: lower bound of regular forward for segment [start, end)
    """
    return meta_data["reg_fwd_low_bound"][(start, end)]

def compute_nograd_forward_lower_bound(meta_data, start, end):
    return meta_data["nograd_fwd_low_bound"][(start, end)]


def compute_budget_lower_bound(meta_data):
    n = len(meta_data["fwd_ts"])
    lower_bound = compute_nograd_forward_lower_bound(meta_data, 0, n)
    return lower_bound

def solver_bottom_up_vectorized(meta_data, budget):
    """
    We define budget here to include gradients, activations and outputs, only excluding inputs
    :param meta_data:
    :param budget:
    :return:
    """
    n = len(meta_data["fwd_ts"])

    dp = np.ones([budget + 1, n + 1, n + 1], dtype=float) * np.inf  # denote budget for segment [i, j)
    # first dimension to denote forward type, second dimension for where to split
    # third dimension for left budget if further divided, forth dimension for right budget
    solution = np.ones([budget + 1, 4, n + 1, n + 1], dtype=int)
    solution[:, 0, ...] = int(ForwardType.Infeasible)
    solution[:, 1, ...] = -1
    solution[:, 2, ...] = -1
    solution[:, 3, ...] = -1
    # solution[:, 4, ...] = 0 # is 1 if left is regular forward
    # parallelize later
    # covers the base case of segm_len == 1
    for start in range(n):
        end = start + 1
        reg_fwd_lo_bound = compute_regular_forward_lower_bound(meta_data, start, end)
        if reg_fwd_lo_bound <= budget:
            dp[reg_fwd_lo_bound:, start, end] = 0
            solution[reg_fwd_lo_bound:, 0, start, end] = int(ForwardType.Regular)
        # no need to assign below, write to be clear
        # dp[:min(regular_forward_lower_bound, budget + 1), start, end] = np.inf
        # solution[:min(regular_forward_lower_bound, budget + 1), 0, start, end] = int(ForwardType.Infeasible)



    for segm_len in tqdm(range(2, n + 1)):
        for start in range(0, n + 1 - segm_len):
            end = start + segm_len
            # base case
            reg_fwd_lo_bound = compute_regular_forward_lower_bound(meta_data, start, end)
            if reg_fwd_lo_bound <= budget:
                dp[reg_fwd_lo_bound:, start, end] = 0
                solution[reg_fwd_lo_bound:, 0, start, end] = int(ForwardType.Regular)
            nograd_fwd_lo_bound = compute_nograd_forward_lower_bound(meta_data, start, end)
            # no need to assign below, write to be clear
            nograd_fwd_lo_bound = min(nograd_fwd_lo_bound, budget + 1)
            # dp[:nograd_fwd_lo_bound, start, end] = np.inf
            # solution[:nograd_fwd_lo_bound, 0, start, end] = int(ForwardType.Infeasible)
            reg_fwd_lo_bound = min(budget + 1, reg_fwd_lo_bound)

            if nograd_fwd_lo_bound >= reg_fwd_lo_bound:
                # safeguard
                continue

            left_budget_offsets_1 = meta_data["left_budget_offset_option1"][(start + 1):end, end]
            right_budget_offsets_1 = meta_data["outputs_mems"][start:(end - 1)]

            time1_1 = np.ones([reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1]) * np.inf
            for i, k in enumerate(range(start + 1, end)):
                offset = left_budget_offsets_1[i]
                time1_1[:, i] = dp[(nograd_fwd_lo_bound - offset):(reg_fwd_lo_bound - offset), start, k]

            time2_1 = np.ones([reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1]) * np.inf
            for i, k in enumerate(range(start + 1, end)):
                offset = right_budget_offsets_1[i]
                time2_1[:, i] = dp[(nograd_fwd_lo_bound - offset):(reg_fwd_lo_bound - offset), k, end]

            # shape of [end - start - 1]
            remat_time_1 = meta_data["remat_time_option1"][start, (start + 1):end]

            # shape of [reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1]
            time_1 = time1_1 + time2_1 + remat_time_1[np.newaxis, :]


            # 2. with grad forward till k
            actual_left_budget_offsets_2 = meta_data["left_budget_offset_option2"][(start + 1):end, end]
            left_budget_offsets_2 = actual_left_budget_offsets_2 + meta_data["reg_fwd_lo_bound_mat"][start, (start + 1):end]
            right_budget_offsets_2 = meta_data["right_budget_offset_option2"][start, (start + 1):end]

            time_2 = np.ones([reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1]) * np.inf

            for i, k in enumerate(range(start + 1, end)):
                left_offset = left_budget_offsets_2[i]
                offset = right_budget_offsets_2[i]
                lo, hi = max((nograd_fwd_lo_bound - left_offset), 0), max((reg_fwd_lo_bound - left_offset), 0)
                if hi - lo > 0:
                    time_2[-(hi-lo):, i] = dp[(lo + left_offset - offset):(hi + left_offset - offset), k, end]

            # argmin of option 1 vs 2
            min_solutions_1 = -np.ones([reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1, 4], dtype=int)
            min_solutions_1[..., 0] = int(ForwardType.Infeasible)
            min_solutions_1[..., 1] = np.arange(start + 1, end)[np.newaxis, :]
            min_solutions_2 = min_solutions_1.copy()

            valid_mask_1 = (time_1 < np.inf)
            min_solutions_1[..., 0][valid_mask_1] = int(ForwardType.Recursive)
            min_solutions_1[..., 2][valid_mask_1] = (np.arange(nograd_fwd_lo_bound, reg_fwd_lo_bound)[:, np.newaxis] - left_budget_offsets_1[np.newaxis, :])[valid_mask_1]
            min_solutions_1[..., 3][valid_mask_1] = (np.arange(nograd_fwd_lo_bound, reg_fwd_lo_bound)[:, np.newaxis] - right_budget_offsets_1[np.newaxis, :])[valid_mask_1]

            valid_mask_2 = (time_2 < np.inf)
            min_solutions_2[..., 0][valid_mask_2] = int(ForwardType.Mixed)
            min_solutions_2[..., 2][valid_mask_2] = \
            (np.arange(nograd_fwd_lo_bound, reg_fwd_lo_bound)[:, np.newaxis] - actual_left_budget_offsets_2[np.newaxis, :])[
                valid_mask_2]
            min_solutions_2[..., 3][valid_mask_2] = \
            (np.arange(nograd_fwd_lo_bound, reg_fwd_lo_bound)[:, np.newaxis] - right_budget_offsets_2[np.newaxis, :])[
                valid_mask_2]

            time_compare = np.array([time_1, time_2]).reshape([2, -1])
            min_options = np.argmin(time_compare, axis=0)
            # shape of [reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1]
            min_options_expand = np.zeros([min_options.shape[0] * 4], dtype=min_options.dtype)
            for i in range(4):
                min_options_expand[i::4] = min_options
            min_ts = np.take_along_axis(time_compare, min_options[np.newaxis, :], axis=0).reshape([reg_fwd_lo_bound - nograd_fwd_lo_bound, -1])
            solutions_compare = np.array([min_solutions_1, min_solutions_2]).reshape([2, -1])
            # shape of [reg_fwd_lo_bound - nograd_fwd_lo_bound, end - start - 1, 4]
            min_solutions = np.take_along_axis(solutions_compare, min_options_expand[np.newaxis, :], axis=0).reshape([reg_fwd_lo_bound - nograd_fwd_lo_bound, -1, 4])

            # argmin over k
            # shape of [end - start - 1]
            min_ks = np.argmin(min_ts, axis=1)
            # shape of [reg_fwd_lo_bound - nograd_fwd_lo_bound]
            min_ts = np.take_along_axis(min_ts, np.expand_dims(min_ks, axis=-1), axis=-1)[..., 0]
            min_ks_expand = np.zeros([min_ks.shape[0] * 4], dtype=min_ks.dtype)
            for i in range(4):
                min_ks_expand[i::4] = min_ks
            min_solutions = np.take_along_axis(min_solutions.transpose(0, 2, 1).reshape(-1, end - start - 1), np.expand_dims(min_ks_expand, axis=-1), axis=-1).reshape(-1, 4)
            dp[nograd_fwd_lo_bound:reg_fwd_lo_bound, start, end] = min_ts
            solution[nograd_fwd_lo_bound:reg_fwd_lo_bound, :, start, end] = min_solutions

    return dp, solution

def parse_solution(dp, solution):
    n = dp.shape[1] - 1
    budget = dp.shape[0] - 1

    root = build_solution_tree(dp, solution, 0, n, budget, root=None)
    return root

def build_solution_tree(dp, solution, start, end, budget, root=None):
    forward_type, middle, left_budget, right_budget = solution[budget, :, start, end]
    min_time = dp[budget, start, end]
    node = Node("segm_{}_{}".format(start, end), start=start, end=end, budget=budget, middle=middle, min_time=min_time, forward_type=forward_type)
    if root is not None:
        node.parent = root
    if forward_type in [int(ForwardType.Regular), int(ForwardType.Infeasible)]:
        return node
    if forward_type == int(ForwardType.Mixed):
        left_node = Node("segm_{}_{}".format(start, middle), start=start, end=middle, budget=left_budget, middle=None, min_time=0, forward_type=int(ForwardType.Regular))
        left_node.parent = node
        build_solution_tree(dp, solution, middle, end, right_budget, root=node)
    elif forward_type == int(ForwardType.Recursive):
        build_solution_tree(dp, solution, start, middle, left_budget, root=node)
        build_solution_tree(dp, solution, middle, end, right_budget, root=node)
    else:
        raise NotImplementedError

    return node

def parse_solution_seq_exec(dp, solution):
    n = dp.shape[1] - 1
    budget = dp.shape[0] - 1

    root = build_solution_tree(dp, solution, 0, n, budget, root=None)
    exec_seq = get_checkpointx_sequential_exec_sequence(root, exec_seq=[])
    return exec_seq

def get_checkpointx_backward_exec_sequence(solution, exec_seq=[]):
    assert solution.forward_type != ForwardType.Infeasible
    start, end = solution.start, solution.end
    if solution.forward_type == int(ForwardType.Regular):
        # conduct normal forward and backward
        exec_seq.append([BackwardType.Normal, start, end])
        return exec_seq

    for segm_solution in solution.children[::-1]:
        seg_start, seg_end = segm_solution.start, segm_solution.end
        if seg_start != start:
            # conduct inputs rematerialization
            exec_seq.append([BackwardType.Checkpoint, start, seg_start])

        exec_seq = get_checkpointx_backward_exec_sequence(segm_solution, exec_seq=exec_seq)

    return exec_seq

def get_checkpointx_sequential_exec_sequence(solution, exec_seq=[]):
    if solution.forward_type == int(ForwardType.Infeasible):
        raise Exception("Unable to find a valid solution")
    elif solution.forward_type == int(ForwardType.Regular):
        # resort to normal forward, backward
        start, end = solution.start, solution.end
        exec_seq.append([RunType.Normal, start, end])
        return exec_seq
    elif solution.forward_type == int(ForwardType.Mixed):
        assert len(solution.children) == 2
        exec_seq = get_checkpointx_sequential_exec_sequence(solution.children[0], exec_seq=exec_seq)
        return get_checkpointx_sequential_exec_sequence(solution.children[1], exec_seq=exec_seq)

    elif solution.forward_type == int(ForwardType.Recursive):
        assert len(solution.children) == 2
        # todo: only the left tree will be wrapped with CheckpointX RunType
        backward_exec_seq = get_checkpointx_backward_exec_sequence(solution.children[0], exec_seq=[])
        start, end = solution.children[0].start, solution.children[0].end
        exec_seq.append([RunType.Checkpointx, start, end, backward_exec_seq])

        return get_checkpointx_sequential_exec_sequence(solution.children[1], exec_seq=exec_seq)
    else:
        raise NotImplementedError