import argparse
import os

import torch
import yaml
from diffusers import FluxPipeline

from toma.flux_scheduler import FluxScheduler
from toma.patch import apply_patch


def get_next_image_index(output_folder):
    """
    Returns the next available index for image files in the output folder.
    Looks for files starting with an integer followed by an underscore.
    """
    if not os.path.exists(output_folder):
        return 1
    files = os.listdir(output_folder)
    indices = []
    for fname in files:
        if fname.endswith(".jpeg"):
            parts = fname.split("_", 1)
            if parts and parts[0].isdigit():
                indices.append(int(parts[0]))
    if indices:
        return max(indices) + 1
    else:
        return 1


def generate_image(
    pipeline,
    output_folder,
    prompt,
    ratio=0,
    random_seed=864,
    dst_selection="local_tile_wise_facility",
    height=1024,
    width=1024,
    num_tiles=256,
    merge_method="attention",
    num_inference_steps=35,
):
    """Generate a single image with the given parameters."""
    print(f"Generating image: {prompt}")
    print(
        f"Parameters: ratio={ratio}, dst_selection={dst_selection}, seed={random_seed}"
    )

    # Initialize scheduler
    recompute_step = list(range(0, num_inference_steps))
    merge_step = list(range(0, num_inference_steps, 3))

    flux_scheduler = FluxScheduler(
        timesteps=num_inference_steps,
        dst_recompute_timesteps=recompute_step,
        attn_recompute_timesteps=recompute_step,
        merge_step=merge_step,
        config_path="./config/flux.yaml",
    )

    # Apply patch if ratio > 0
    if ratio > 0:
        apply_patch(
            pipeline,
            ratio=ratio,
            dst_selection=dst_selection,
            num_tiles=num_tiles,
            merge_method=merge_method,
            merge_scheduler=flux_scheduler,
        )

    # Generate image
    generator = torch.Generator(device=pipeline.device).manual_seed(random_seed)

    output, _ = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
    )

    image = output.images[0]

    # Save image
    os.makedirs(output_folder, exist_ok=True)

    # Determine the image index automatically
    image_index = get_next_image_index(output_folder)

    file_name = f"{prompt[:30]}_{random_seed}_{ratio}.jpeg"
    file_name = file_name.replace(" ", "_")
    indexed_file_name = f"{image_index}_{file_name}"
    image_path = os.path.join(output_folder, indexed_file_name)

    image.save(image_path, "JPEG")

    print(f"Image saved: {image_path}")
    return image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with FLUX pipeline")
    parser.add_argument("--prompt", type=str, default="A photo of a tench, a type of fish", help="Text prompt for image generation")
    parser.add_argument("--ratio", type=float, default=0.5, help="Token merging ratio (0-1)")
    parser.add_argument("--seed", type=int, default=864, help="Random seed")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize pipeline
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(args.device)

    pipeline.set_progress_bar_config(disable=False)

    # Fill in the prompts_list with some example ImageNet-1k class prompts.
    prompts_list = [
        "A photo of a tench, a type of fish",
        "A photo of a goldfish",
        "A hyper-realistic portrait of a great white shark breaching the ocean surface, water droplets sparkling in the golden sunrise, cinematic lighting, ultra-detailed, by Greg Rutkowski and Artgerm",
        "A majestic tiger shark gliding through crystal clear tropical waters, surrounded by vibrant coral reefs and schools of colorful fish, photorealistic, high detail, by Paul Nicklen",
        "A sprawling futuristic cityscape at sunset, neon lights reflecting off glass skyscrapers, flying cars zipping between buildings, bustling streets, cinematic, by Syd Mead and Beeple",
        "A playful cat wearing retro sunglasses, confidently riding a skateboard down a sunlit urban street, dynamic motion blur, whimsical, by Loish and Studio Ghibli",
        "A surreal dreamscape with floating islands covered in lush forests, cascading waterfalls pouring into the clouds, ethereal lighting, fantasy art, by James Gurney and Lisa Frank",
        "A steaming bowl of ramen on a wooden table, with a tiny dragon playfully emerging from the noodles, whimsical, highly detailed, soft warm lighting, by Sachin Teng",
        "A humanoid robot standing before a canvas, painting a self-portrait with expressive brushstrokes, studio lighting, introspective mood, by Simon Stålenhag and Pascal Blanche",
        "An enchanted forest at twilight, glowing bioluminescent mushrooms illuminating the mossy ground, magical atmosphere, misty background, by Eyvind Earle and Brian Froud",
        "A massive steampunk airship soaring above snow-capped mountains, intricate brass machinery, dramatic clouds, golden hour lighting, by Ian McQue and Peter de Sève",
        "A regal dog in ornate medieval knight armor, standing proudly in a grand stone hall, oil painting style, rich textures, by Rembrandt and WLOP",
        "A cozy wooden cabin nestled in a snowy forest at night, warm light glowing from the windows, smoke curling from the chimney, peaceful winter scene, by Thomas Kinkade",
        "A vibrant underwater coral reef teeming with colorful fish, sea turtles, and swaying anemones, sunbeams filtering through the water, ultra-detailed, by David Doubilet",
    ]
    for prompt in prompts_list:
        # Generate image
        generate_image(
            pipeline=pipeline,
            output_folder=args.output,
            prompt=prompt,
            ratio=args.ratio,
            random_seed=args.seed,
            height=args.height,
            width=args.width,
            num_tiles=config.get("num_tiles", 256),
            num_inference_steps=config.get("num_of_inference_steps", 35),
        )
