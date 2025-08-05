import torch
from .utils import fold_with_indices


def stripe_attention_merge(
    x: torch.Tensor,
    stripe_idx_stacked: torch.Tensor,
    num_of_tiles: int,
    attention_scale: float = 1000.0,
) -> torch.Tensor:
    """
    Computes the attention weights for stripe-wise merging.
    Used for global stripe-wise facility location (text processing).

    Args:
        x: Input tensor
        stripe_idx_stacked: Stripe indices
        num_of_tiles: Number of tiles
        attention_scale: Scaling factor for attention computation (default: 10000.0)
    """
    B, N, C = x.shape
    k = num_of_tiles

    x_stacked = x.reshape(B * k, -1, C)

    dst_stacked = torch.gather(x, 1, stripe_idx_stacked.expand(-1, -1, C)).reshape(
        B * k, -1, C
    )
    A = dst_stacked @ x_stacked.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A * attention_scale, dim=-2)
    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]

    return avg_A, A.transpose(-1, -2)


def tile_attention_merge(
    x: torch.Tensor,
    dst_idx: torch.Tensor,
    num_tiles: int,
    attention_scale: float = 1000.0,
) -> torch.Tensor:
    """
    Computes the attention weights for tile-wise merging.
    Used for local tile-wise facility location (image processing).

    Args:
        x: Input tensor
        dst_idx: Destination indices
        num_tiles: Number of tiles
        attention_scale: Scaling factor for attention computation (default: 100000.0)
    """
    x_reshaped, flatten_idx = fold_with_indices(x, num_tiles)
    dst = torch.take_along_dim(x_reshaped, dst_idx.unsqueeze(-1), dim=2)

    Q = x_reshaped
    K = dst

    # Compute attention scores and normalize
    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A * attention_scale, dim=-2)

    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]]
    A_inv = A.transpose(-1, -2)

    return avg_A, A_inv, flatten_idx

    # Below is an equal implementation using scaled_dot_product_attention
    # Showcasing the possibility of using SDPA for merge computation

    from torch.nn.functional import scaled_dot_product_attention as sdpa

    Q = x_reshaped.squeeze(0)
    K = dst.squeeze(0)

    if not hasattr(tile_attention_merge, "V"):
        tile_attention_merge.V = (
            torch.eye(K.shape[1], device=K.device, dtype=K.dtype)
            .unsqueeze(0)
            .expand(K.shape[0], -1, -1)
            .contiguous()
        )  # [B, num_dst, num_dst]
    A = sdpa(Q, K, tile_attention_merge.V, scale=attention_scale).unsqueeze(0)

    count_per_dst = A.sum(dim=-2, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]]

    return avg_A.transpose(-1, -2), A, flatten_idx
