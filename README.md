<h1 align="center">ToMA</h1>

![ToMA](./assets/first_illustration.jpg)

This is the official implementation of our paper:  
**[Token Merging with Attention for Diffusion Models](https://icml.cc/virtual/2025/poster/46449)**  
[Wenbo Lu](https://github.com/wenbolu), [Shaoyi Zheng](https://github.com/zhengshaoyi), [Yuxuan Xia](https://github.com/xyxuan), [Shengjie Wang](https://github.com/wangshengjie-ai)  
_[ICML2025 Poster](https://icml.cc/virtual/2025/poster/46449)_ | _[Paper](https://icml.cc/virtual/2025/poster/46449)_ | _[BibTeX](#citation)_ 

ToMA reformulates token merging as a _linear transformation_, enabling a natural implementation within the attention mechanism. Guided by the theoretical guarantees of submodular optimization, our method achieves faster inference with minimal image degradation.


## 🚀 Installation

We recommend using [Miniforge](https://github.com/conda-forge/miniforge) instead of the default Anaconda/Miniconda to quickly resolve dependency issues. After installing Miniforge, set up the environment with:

```bash
mamba env create -f environment.yaml
mamba activate toma_env
```

**Note:** We have observed that certain versions of PyTorch can be unexpectedly slow for this project. Please use the versions specified in `environment.yaml` for best performance.

## 📝 Usage

For a typical workflow for generating images with ToMA and the Flux pipeline, refer to `run_flux.py` and the docstrings throughout the codebase.  
You can configure merge scheduler and settings via the `config/config.yaml` file, which allows you to easily adjust parameters such as the number of tiles, number of merge steps, and other options to control merge process.


## 🕸️ Pipeline Overview

```mermaid
graph TD
    %% Initialization Phase
    start([Start Image Generation]) --> run_flux[run_flux.py]
    run_flux --> init_scheduler[Initialize FluxScheduler]
    run_flux --> init_model[Initialize FluxPipeline Model]
    
    %% Patching Phase
    init_scheduler --> check_ratio{ratio > 0?}
    check_ratio -->|Yes| apply_patch[apply_patch in patch.py]
    check_ratio -->|No| flow_matching[Flow Matching Process]
    
    apply_patch --> replace_blocks[Replace Original Blocks with Customized Blocks]
    apply_patch --> custom_attn[Use Customized Attention Processor]
    replace_blocks --> flow_matching
    custom_attn --> flow_matching
    
    %% Flow Matching and Denoising Process
    flow_matching --> denoising_loop[Denoising Loop]
    denoising_loop --> scheduler_step[FluxScheduler.step_for_image]
    scheduler_step --> check_merge{Should Merge at This Step?}
    
    %% Merge Decision and Process
    check_merge -->|Yes| compute_merge[Compute How to Merge]
    check_merge -->|No| next_step[Continue to Next Step]
    
    compute_merge --> facility_location[Facility Location Algorithm]
    facility_location --> select_dest[Select Destination Tokens]
    select_dest --> token_mapping[Map All Tokens to Destinations]
    
    %% Token Processing
    token_mapping --> merge_down[Merge Down Tokens]
    merge_down --> handle_rotary[Handle Rotary Embeddings]
    handle_rotary --> apply_merge[Apply Merge Transformation]
    
    %% Unmerge Process
    apply_merge --> unmerge[Unmerge to Restore Original Size]
    unmerge --> next_step
    
    %% Continue or Finish
    next_step --> more_steps{More Denoising Steps?}
    more_steps -->|Yes| denoising_loop
    more_steps -->|No| save_image[Save Generated Image]
    
    %% Final Output
    save_image --> done[Done]
    
    %% Subprocesses
    subgraph "Facility Location Methods"
        local_tile[local_tile_wise_facility]
        stripe_wise[stripe_wise_facility]
        global_stripe[global_stripe_wise_facility]
    end
    
    subgraph "Merge Methods"
        tile_attn[tile_attention_merge]
        stripe_attn[stripe_attention_merge]
    end
    
    subgraph "Core Components"
        flux_attn[flux_attention_processor.py]
        merge_utils[merge.py]
        merge_methods[merge_methods.py]
        facility[facility_location.py]
        utils[utils.py]
    end
    
    %% Connections to subprocesses
    facility_location --> local_tile
    facility_location --> stripe_wise
    facility_location --> global_stripe
    
    apply_merge --> tile_attn
    apply_merge --> stripe_attn
    
    %% Component dependencies
    apply_patch --> flux_attn
    apply_patch --> merge_utils
    apply_patch --> utils
    
    flux_attn --> merge_utils
    flux_attn --> utils
    
    merge_utils --> merge_methods
    merge_utils --> facility
    merge_utils --> utils
    
    merge_methods --> utils
    
    %% Styling
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef component fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef subprocess fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class run_flux,init_scheduler,init_model,apply_patch,replace_blocks,custom_attn,flow_matching,denoising_loop,scheduler_step,compute_merge,facility_location,select_dest,token_mapping,merge_down,handle_rotary,apply_merge,unmerge,next_step,save_image process
    class check_ratio,check_merge,more_steps decision
    class flux_attn,merge_utils,merge_methods,facility,utils component
    class local_tile,stripe_wise,global_stripe,tile_attn,stripe_attn subprocess
```

## Acknowledgements

We gratefully acknowledge the support of the **NYU HPC** team for providing computational resources and technical assistance that made this work possible.


## Citation

If you use ToMA in your work, please cite:

```bibtex
@inproceedings{
lu2025toma,
title={To{MA}: Token Merge with Attention for Diffusion Models},
author={Wenbo Lu and Shaoyi Zheng and Yuxuan Xia and Shengjie Wang},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=51l8tvuIxo}
}
```
