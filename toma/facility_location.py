import torch


def batched_facility_location(x, r):
    """
    Find the top r representatives (for dst) for the batch using the facility location algorithm.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size,
            N is the number of tokens, and C is the latent dimension.
        r (int): The number of representatives to select.

    Returns:
        torch.Tensor: Tensor of shape (B, r) containing the indices of the selected representatives.

    Raises:
        AssertionError: If r is not within the range (0, N].
    """
    B, N, C = x.shape
    assert 0 < r <= N, "r should be within the range (0, N]"
    device = x.device
    x = x.float()

    import torch.nn.functional as F

    x = F.normalize(x, p=2, dim=-1)

    similarity_matrix = x @ x.transpose(-1, -2)

    # Sum similarity matrix rows and initialize first representative
    row_sums = torch.sum(similarity_matrix, dim=2)
    init_v = torch.argmax(row_sums, dim=1)

    max_sim = torch.gather(
        similarity_matrix, 1, init_v.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, N)
    ).squeeze(1)

    # Initialize representatives tensor
    representatives = torch.zeros(
        B, r, dtype=torch.long, device=similarity_matrix.device
    )
    representatives[:, 0] = init_v  # Set initial representatives

    # Loop to find the remaining representatives using facility location
    for i in range(1, r):
        # Expand max_sim to match dimensions for broadcasting
        expanded_max_sim = max_sim.unsqueeze(1).expand(B, N, N)

        # Compute differences, ensuring non-negative values with ReLU
        differences = torch.relu(similarity_matrix - expanded_max_sim)

        row_sums = torch.sum(differences, dim=2)

        # Find the index of the maximum sum in each batch
        next_v = torch.argmax(row_sums, dim=1)

        # Update representatives and maximum similarity vector
        representatives[:, i] = next_v

        max_sim = torch.max(
            max_sim, similarity_matrix[torch.arange(B, device=device), next_v]
        )
    return representatives


def stripe_wise_facility(x, r, k):
    """
    Divide the original tokens into chunks and perform batched facility location algorithm on each chunk.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size,
            N is the number of tokens, and C is the latent dimension.
        r (int): The number of representatives to select.
        k (int): The number of chunks to divide the tokens into.

    Returns:
        torch.Tensor: Tensor of shape (B, r) containing the indices of the selected representatives.
    """
    B, N, C = x.shape
    assert 0 < r <= N, "r should be within the range (0, N]"
    # padding 0 to make N divisible by k
    if N % k != 0:
        print("padding")
        pad = k - N % k
        x = torch.cat([x, torch.zeros(B, pad, C, device=x.device)], dim=1)

    chunk_size = N // k
    r_per_chunk = r // k

    chunked_x = x.view(k, chunk_size, C)

    # batched facility location algorithm
    stacked_representatives = batched_facility_location(chunked_x, r_per_chunk)
    stacked_representatives = torch.sort(stacked_representatives, dim=1).values
    representatives = stacked_representatives.view(1, k * r_per_chunk)

    # remap the indices to the original tokens
    remap_offset = (
        torch.arange(0, k, device=x.device).repeat_interleave(r_per_chunk).unsqueeze(0)
        * chunk_size
    )
    representatives = (
        representatives + remap_offset
    )  # offset by (0,0,0, chunk_size, chunk_size, chunk_size, ...)
    return representatives


def local_tile_wise_facility(x, r, num_patches):
    """
    Divide the original tokens into regional patches and perform batched facility location algorithm on each chunk.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H*W, C)
        r (int): The number of representatives to select
        num_patches (int): The number of patches to split the tensor into.

    Returns:
        torch.Tensor: Tensor of shape (B, num_patches, r_per_patch) containing the indices of the selected representatives.
    """
    B, HW, C = x.shape
    assert 0 < r <= HW, "r should be within the range (0, N]"
    x = x[0].unsqueeze(0)  # modify for batch 1

    H = W = int(HW**0.5)  # Calculate H and W assuming a square image
    r_per_patch = r // num_patches  # Number of representatives per patch

    # Reshape the tensor to B, H, W, C
    x = x.view(1, H, W, C)
    # Calculate size of each patch
    patch_side_len = H // int(num_patches**0.5)

    # unfold along height and width
    x = x.permute(0, 3, 1, 2)
    patches = x.unfold(2, patch_side_len, patch_side_len).unfold(
        3, patch_side_len, patch_side_len
    )
    # Reshape to combine patches into the batch dimension
    stacked_patches = (
        patches.contiguous()
        .view(1, C, -1, patch_side_len * patch_side_len)
        .permute(0, 2, 3, 1)
    )

    # batched facility location algorithm
    stacked_representatives = batched_facility_location(
        stacked_patches.reshape(num_patches, patch_side_len**2, C), r_per_patch
    ).reshape(1, num_patches, r_per_patch)
    stacked_representatives = stacked_representatives.repeat(B, 1, 1)

    return stacked_representatives
