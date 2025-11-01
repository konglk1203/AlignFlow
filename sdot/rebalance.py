import torch
from collections import defaultdict


def rebalance(tensor, num_targets):
    """
    Remaps integers in a tensor to achieve a more uniform frequency distribution,
    minimizing the number of changed elements.

    See Rmk. ?? in the paper for details.

    Args:
        tensor (torch.Tensor): A 1D tensor of integers (0 to n-1).

    Returns:
        torch.Tensor: A new tensor with remapped integers, and the number of changes made.
    """

    if not torch.is_tensor(tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional.")
    if tensor.dtype not in [torch.int32, torch.int64]:
        raise ValueError("Input tensor must contain integer types.")

    # 1. Calculate Frequencies
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    current_freqs = {
        val.item(): count.item() for val, count in zip(unique_elements, counts)
    }

    # Ensure all integers from 0 to max_val are considered, even if some have 0 frequency
    max_val = num_targets - 1
    all_possible_values = torch.arange(max_val + 1).tolist()

    # Initialize frequencies for all possible values, including those not present
    full_freqs = {i: current_freqs.get(i, 0) for i in all_possible_values}

    num_elements = tensor.numel()
    num_unique_possible = len(all_possible_values)

    if num_unique_possible == 0:
        return tensor.clone(), 0  # Empty tensor, no changes

    # 2. Determine Target Frequency (average frequency)
    target_frequency = num_elements / num_unique_possible

    # 3. Identify "Excess" and "Deficit"
    excess_values = defaultdict(int)  # {value: excess_count}
    deficit_values = defaultdict(int)  # {value: deficit_count}

    for val, count in full_freqs.items():
        if count > target_frequency:
            excess_values[val] = int(
                count - target_frequency
            )  # Convert to int for easier handling
        elif count < target_frequency:
            deficit_values[val] = int(target_frequency - count)

    # 4. Pair Excess with Deficit (Greedy Approach)
    # Sort by amount (descending for excess, ascending for deficit - but we consume from largest deficit)
    sorted_excess_items = sorted(
        excess_values.items(), key=lambda item: item[1], reverse=True
    )
    sorted_deficit_items = sorted(
        deficit_values.items(), key=lambda item: item[1], reverse=True
    )  # Also largest deficit first

    remapped_tensor = tensor.clone()
    total_changes = 0

    # Create a list of indices for each value for efficient modification
    indices_map = {
        val: (tensor == val).nonzero(as_tuple=True)[0]
        for val in unique_elements.tolist()
    }

    # Track which indices have already been changed
    changed_indices = torch.zeros_like(tensor, dtype=torch.bool)

    i_excess = 0
    i_deficit = 0

    while i_excess < len(sorted_excess_items) and i_deficit < len(sorted_deficit_items):
        excess_val, current_excess_amount = sorted_excess_items[i_excess]
        deficit_val, current_deficit_amount = sorted_deficit_items[i_deficit]

        if current_excess_amount <= 0:  # This excess source is depleted
            i_excess += 1
            continue
        if current_deficit_amount <= 0:  # This deficit target is filled
            i_deficit += 1
            continue

        if excess_val == deficit_val:  # Don't try to map a value to itself
            i_excess += 1
            i_deficit += 1
            continue

        # Determine how many elements to move in this step
        num_to_change = min(current_excess_amount, current_deficit_amount)

        # Find available indices of excess_val that haven't been changed yet
        available_excess_indices = indices_map.get(
            excess_val, torch.empty(0, dtype=torch.long)
        )

        # Filter out indices that have already been changed
        unmodified_excess_indices = available_excess_indices[
            ~changed_indices[available_excess_indices]
        ]

        if unmodified_excess_indices.numel() == 0:
            i_excess += 1
            continue

        # Take the first `num_to_change` available indices
        indices_to_modify = unmodified_excess_indices[:num_to_change]

        if indices_to_modify.numel() == 0:
            i_excess += 1  # No more items of this excess_val to change
            continue

        # Perform the change
        remapped_tensor[indices_to_modify] = deficit_val
        changed_indices[indices_to_modify] = True
        total_changes += indices_to_modify.numel()

        # Update remaining amounts
        sorted_excess_items[i_excess] = (
            excess_val,
            current_excess_amount - indices_to_modify.numel(),
        )
        sorted_deficit_items[i_deficit] = (
            deficit_val,
            current_deficit_amount - indices_to_modify.numel(),
        )

    return remapped_tensor, total_changes
