from typing import Callable, Tuple

import torch
from .facility_location import local_tile_wise_facility, stripe_wise_facility
from .merge_methods import tile_attention_merge, stripe_attention_merge
from .utils import do_nothing, fold_with_indices, unfold_with_indices


def merge_helper(
    x: torch.Tensor,
    r: int,
    num_of_tiles: int = 64,
    dst_selection: str = "original",
    merge_scheduler=None,
    key_word=None,
    rope_emb=None,
    attention_scale: float = 1000.0,
) -> Tuple[Callable, Callable]:

    if key_word == "image":
        if_recompute_attn, if_not_merge = merge_scheduler.step_for_image()
    elif key_word == "text":
        if_recompute_attn, if_not_merge = merge_scheduler.step_for_text()
        dst_selection = "global_stripe_wise_facility"
        return False, False

    if if_not_merge:
        return False, False

    B, N, C = x.shape
    x = x / x.norm(dim=-1, keepdim=True)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        num_dst = N - r

        def select_destination():
            if dst_selection == "local_tile_wise_facility":
                dst_idx = (
                    local_tile_wise_facility(x[0].unsqueeze(0), num_dst, num_of_tiles).repeat(B, 1, 1).to(x.device)
                )
            elif dst_selection == "global_stripe_wise_facility":
                dst_idx = stripe_wise_facility(x[0].unsqueeze(0), num_dst, num_of_tiles).repeat(B, 1).to(x.device)
            else:
                raise ValueError(f"Unknown dst_selection: {dst_selection}")

            return dst_idx

        if if_recompute_attn:
            if dst_selection == "local_tile_wise_facility":
                dst_idx = merge_scheduler.get_dst_idx_general(select_destination, key_word).to(x.device)
                # FIX: Remove attention_scale as a keyword argument, pass as positional
                A, A_inv, flatten_idx = merge_scheduler.get_A_general(
                    tile_attention_merge,
                    key_word,
                    True,
                    x,
                    dst_idx,
                    num_of_tiles,
                    attention_scale,
                )
                rope_emb, _ = fold_with_indices(rope_emb, num_of_tiles)
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=2,
                    index=dst_idx.repeat(2, 1, 1).unsqueeze(-1).expand(-1, -1, -1, rope_emb.size(-1)),
                ).to(x.device)
                image_rotary_emb = image_rotary_emb.reshape(2, -1, rope_emb.size(-1))
                image_rotary_emb = merge_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
            elif dst_selection == "global_stripe_wise_facility":
                dst_idx = merge_scheduler.get_dst_idx_general(select_destination, key_word)
                # FIX: Remove attention_scale as a keyword argument, pass as positional
                A, A_inv = merge_scheduler.get_A_general(
                    stripe_attention_merge,
                    key_word,
                    False,
                    x,
                    dst_idx,
                    num_of_tiles,
                    attention_scale,
                )
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=1,
                    index=dst_idx.expand(2, -1, rope_emb.size(-1)),
                )
                image_rotary_emb = merge_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
        else:
            dst_idx = merge_scheduler.get_dst_idx_general(do_nothing, key_word, x)
            if dst_selection == "local_tile_wise_facility":
                A, A_inv, flatten_idx = merge_scheduler.get_A_general(do_nothing, key_word, True, x)
            else:
                A, A_inv = merge_scheduler.get_A_general(do_nothing, key_word, False, x)
            image_rotary_emb = merge_scheduler.get_rope_emb_general(None, key_word)

    def merge(x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, x), dst_idx, image_rotary_emb

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A_inv, x)

    if dst_selection == "local_tile_wise_facility":

        def tile_wise_merge(x: torch.Tensor) -> torch.Tensor:
            x_reshaped, _ = fold_with_indices(x, num_of_tiles)
            x_merged = A @ x_reshaped
            x_merged = x_merged.reshape(B, -1, C)
            return x_merged.reshape(B, -1, C), dst_idx, image_rotary_emb

        def tile_wise_unmerge(x: torch.Tensor) -> torch.Tensor:
            num_tiles = A_inv.shape[1]
            x = x.reshape(B, num_tiles, -1, C)
            res = A_inv @ x
            unfold_x = unfold_with_indices(res, flatten_idx)
            return unfold_x

        return tile_wise_merge, tile_wise_unmerge

    return merge, unmerge
