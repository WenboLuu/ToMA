import yaml
from typing import Callable


class FluxScheduler:
    def __init__(
        self,
        timesteps: int,
        dst_recompute_timesteps: list,
        attn_recompute_timesteps: list,
        merge_step: list,
        config_path: str,
    ):
        assert set(dst_recompute_timesteps).issubset(
            set(attn_recompute_timesteps)
        ), "dst_recompute_timesteps should be subset of attn_recompute_timesteps"

        self.dst_recompute_t = dst_recompute_timesteps
        self.attn_recompute_t = attn_recompute_timesteps
        self.merge_step = merge_step

        self.model_structure = self._initialize_structure(config_path)
        self.blockidx2name = {}
        accumulated_sum = 0
        # Assigning block name for every index in its range
        for block_name, block_count in self.model_structure.items():
            for idx in range(accumulated_sum, accumulated_sum + block_count):
                self.blockidx2name[idx] = block_name
            accumulated_sum += block_count

        self.timesteps = timesteps
        self.block_cnt = sum(self.model_structure.values())
        self.total_block_counts = self.block_cnt * timesteps
        self.block_counter = -1
        self.block_storage = self._initialize_storage(self.model_structure)
        self.current_timestep = -1
        self.current_block_idx = -1

        self.first_block_idx = self._get_first_block_of_each_type()

    def _get_first_block_of_each_type(self):
        """
        Precompute and return a set of indices representing the first occurrence of each block type.
        """
        first_block_indices = set()
        seen_blocks = set()

        for idx, block_name in self.blockidx2name.items():
            if block_name not in seen_blocks:
                first_block_indices.add(idx)
                seen_blocks.add(block_name)

        return first_block_indices

    def _initialize_storage(self, model_structure: dict) -> dict:
        storage = {}
        for block_name, _ in model_structure.items():
            storage[block_name] = {
                "dst_idx": None,
                "dst_idx_for_prompt": None,
                "dst_idx_for_ff": None,
                "dst_idx_for_text": None,
                "dst_idx_for_image": None,
                "A": None,
                "A_inv": None,
                "A_prompt": None,
                "A_prompt_inv": None,
                "A_ff": None,
                "A_inv_ff": None,
                "A_text": None,
                "A_inv_text": None,
                "A_image": None,
                "A_inv_image": None,
                "image_rotary_emb": None,
                "image_rotary_emb_for_prompt": None,
            }
        return storage

    def _initialize_structure(self, config_path) -> dict:
        """
        Initialize the merge structure from a YAML configuration file.
        """
        with open(config_path, "r") as f:
            structure_config = yaml.safe_load(f)
        # Multiply the number of blocks by the repeat count for each block type
        structure = {
            block_name: details["num_blocks"] * details["repeat_count"]
            for block_name, details in structure_config.items()
        }
        return structure

    def step_for_image(self):
        if self.block_counter >= self.total_block_counts:
            print("All blocks and repetitions have been processed.")
            self.reset()
            return False

        self.block_counter += 1
        self.current_timestep = self.block_counter // self.block_cnt
        self.current_block_idx = self.block_counter % self.block_cnt
        self.current_block_name = self.blockidx2name.get(self.current_block_idx)

        if_recompute_t = self.current_timestep in self.attn_recompute_t
        if_recompute_block = self.current_block_idx in self.first_block_idx
        if_recompute_attn = if_recompute_t and if_recompute_block
        if_not_merge = (
            self.current_block_name == "FluxSingleTransformerBlock_1"
            or self.current_block_name == "FluxJointTransformerBlock"
            or self.current_timestep not in self.merge_step
        )
        return if_recompute_attn, if_not_merge

    def step_for_text(self):
        if self.block_counter >= self.total_block_counts:
            print("All blocks and repetitions have been processed.")
            self.reset()
            return False

        self.current_timestep = self.block_counter // self.block_cnt
        self.current_block_idx = self.block_counter % self.block_cnt
        self.current_block_name = self.blockidx2name.get(self.current_block_idx)

        if_recompute_t = self.current_timestep in self.attn_recompute_t
        if_recompute_block = self.current_block_idx in self.first_block_idx
        if_recompute_attn = if_recompute_t and if_recompute_block
        if_not_merge = (
            self.current_block_name == "FluxSingleTransformerBlock_1"
            or self.current_block_name == "FluxJointTransformerBlock"
            or self.current_timestep not in self.merge_step
        )
        return if_recompute_attn, if_not_merge

    def get_current_t_block(self):
        current_block_name = self.blockidx2name.get(self.current_block_idx)
        if_first_block = self.current_block_idx in self.first_block_idx
        return self.current_timestep, current_block_name, if_first_block

    def get_dst_idx_general(self, compute_fn: Callable, key_word, *args):
        current_timestep, current_block, if_first_block = self.get_current_t_block()

        if current_timestep in self.dst_recompute_t and if_first_block:
            self.block_storage[current_block][f"dst_idx_for_{key_word}"] = compute_fn(*args)
        return self.block_storage[current_block][f"dst_idx_for_{key_word}"]

    def get_rope_emb_general(self, image_rotary_emb, key_word):
        current_timestep, current_block, if_first_block = self.get_current_t_block()
        if current_timestep in self.dst_recompute_t and if_first_block:
            # if current_timestep in self.dst_recompute_t:

            if key_word == "text":
                self.block_storage[current_block][f"image_rotary_emb_for_prompt"] = image_rotary_emb
            elif key_word == "image":
                self.block_storage[current_block]["image_rotary_emb"] = image_rotary_emb
        return (
            self.block_storage[current_block][f"image_rotary_emb_for_prompt"]
            if key_word == "text"
            else self.block_storage[current_block]["image_rotary_emb"]
        )

    def get_A_general(self, compute_fn: Callable, key_word, require_flat_idx=False, *args):
        current_timestep, current_block, if_first_block = self.get_current_t_block()
        if current_timestep in self.attn_recompute_t and if_first_block:
            if require_flat_idx:
                (
                    self.block_storage[current_block][f"A_{key_word}"],
                    self.block_storage[current_block][f"A_inv_{key_word}"],
                    self.block_storage[current_block]["flat_idx"],
                ) = compute_fn(*args)
            else:
                (
                    self.block_storage[current_block][f"A_{key_word}"],
                    self.block_storage[current_block][f"A_inv_{key_word}"],
                ) = compute_fn(*args)
        if require_flat_idx:
            return (
                self.block_storage[current_block][f"A_{key_word}"],
                self.block_storage[current_block][f"A_inv_{key_word}"],
                self.block_storage[current_block]["flat_idx"],
            )
        else:
            return (
                self.block_storage[current_block][f"A_{key_word}"],
                self.block_storage[current_block][f"A_inv_{key_word}"],
            )

    def reset(self):
        self.block_counter = 0
        self.block_storage = self._initialize_storage(self.model_structure)
