import torch
from typing import Union, Tuple


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def do_nothing(x: torch.Tensor, mode: str = None):
    """Identity function that returns the input unchanged."""
    return x


def mps_gather_workaround(input, dim, index):
    """Workaround for gather operation on MPS devices."""
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def fold_with_indices(x, num_tiles):
    """
    Fold the input tensor into tiles for tile-wise processing.
    Used for local tile-wise facility location (image processing).
    """
    B, N, C = x.shape
    H = W = int(N**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    idx = torch.arange(N, device=x.device, dtype=torch.long)
    idx = idx.reshape(1, H, W, 1)

    tile_idx = torch.as_strided(
        idx,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
        (N, tile_side_len * W, tile_side_len, W, 1, 1),
    )

    flatten_tile_idx = tile_idx.reshape(1, N, 1).expand(B, -1, C)

    flatten_tile_x = torch.gather(x, 1, flatten_tile_idx)
    tile_x = flatten_tile_x.reshape(B, num_tiles, -1, C)

    return tile_x, flatten_tile_idx


def unfold_with_indices(x_prime, flatten_tile_idx):
    """
    Unfold the tiled tensor back to original shape.
    Used for local tile-wise facility location (image processing).
    """
    B, N, C = flatten_tile_idx.shape

    x_prime = x_prime.reshape(B, -1, C)

    restored_x = torch.zeros_like(flatten_tile_idx, dtype=x_prime.dtype)

    restored_x = restored_x.scatter(1, flatten_tile_idx, x_prime)

    return restored_x


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
