from typing import Type

import torch
from .flux_attention_processor import FluxGlobalAttentionProcessor, compute_merge
from .merge import do_nothing
from .utils import isinstance_str


def create_flux_transformer_block(
    block_class: Type[torch.nn.Module],
) -> Type[torch.nn.Module]:
    class TomaBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )

            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)
            joint_attention_kwargs = joint_attention_kwargs or {}

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
                + c_shift_mlp[:, None]
            )

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states

    return TomaBlock


def create_flux_single_transformer_block(
    block_class: Type[torch.nn.Module],
) -> Type[torch.nn.Module]:
    # README: This version merge attention only, do not merge MLP.
    class TomaBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):

            original_embedding = image_rotary_emb

            # RoPE splitting for text and image
            image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb
            rotary_emb_image_0 = image_rotary_emb_1[512:, :]
            rotary_emb_image_1 = image_rotary_emb_2[512:, :]

            rotary_emb_text_0 = image_rotary_emb_1[:512, :]
            rotary_emb_text_1 = image_rotary_emb_2[:512, :]

            cos_sin_image = torch.stack([rotary_emb_image_0, rotary_emb_image_1])
            cos_sin_text = torch.stack([rotary_emb_text_0, rotary_emb_text_1])

            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            hidden_states_copy = norm_hidden_states.clone()

            text_hidden_states = norm_hidden_states[:, :512, :]
            image_hidden_states = norm_hidden_states[:, 512:, :]

            merge_image, unmerge_image = compute_merge(
                image_hidden_states, "image", cos_sin_image, self._toma_info
            )

            if merge_image == False:
                merge_image = do_nothing
                unmerge_image = do_nothing
            else:
                image_hidden_states, dst_idx, image_rotary_emb = merge_image(
                    image_hidden_states
                )

            text_len = 512
            text_rotary_emb = cos_sin_text

            if merge_image == do_nothing:
                rotary_emb = original_embedding
            else:
                rotary_emb = torch.cat([text_rotary_emb, image_rotary_emb], dim=1)

            norm_hidden_states = torch.cat(
                [text_hidden_states, image_hidden_states], dim=1
            )

            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=rotary_emb,
                **joint_attention_kwargs,
            )

            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = self.proj_out(hidden_states)

            text_hidden_states = hidden_states[:, :text_len, :]
            image_hidden_states = hidden_states[:, text_len:, :]

            image_hidden_states = unmerge_image(image_hidden_states)
            hidden_states = torch.cat([text_hidden_states, image_hidden_states], dim=1)

            hidden_states = gate * hidden_states
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states

    return TomaBlock


def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    dst_selection: str = "original",
    num_tiles: int = 16,
    merge_method: str = "original",
    merge_scheduler=None,
    attention_scale: float = 1000.0,
):
    remove_patch(model)

    is_diffusers_flux = isinstance_str(model, "FluxPipeline")

    if is_diffusers_flux:
        transformer_model = model.transformer
    else:
        print("Model is not a supported model for Toma patching.")

    transformer_model._toma_info = {
        "size": None,
        "args": {
            "ratio": ratio,
            "dst_selection": dst_selection,
            "num_tiles": num_tiles,
            "merge_method": merge_method,
            "merge_scheduler": merge_scheduler,
            "attention_scale": attention_scale,
        },
    }

    make_toma_block_fn = create_flux_transformer_block
    make_single_toma_block_fn = create_flux_single_transformer_block

    for _, module in transformer_model.named_modules():
        if isinstance_str(module, "FluxTransformerBlock"):
            module.__class__ = make_toma_block_fn(module.__class__)
            module._toma_info = transformer_model._toma_info
            module.attn.processor = FluxGlobalAttentionProcessor()
            module.attn.processor._toma_info = module._toma_info
        elif isinstance_str(module, "FluxSingleTransformerBlock"):
            module.__class__ = make_single_toma_block_fn(module.__class__)
            module._toma_info = transformer_model._toma_info
    return model


def remove_patch(model: torch.nn.Module):
    """Removes a patch from a Toma Diffusion module if it was already patched."""
    # For diffusers
    model = model.transformer

    for _, module in model.named_modules():
        if module.__class__.__name__ == "TomaBlock":
            module.__class__ = module._parent

    return model
