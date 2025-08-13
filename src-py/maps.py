import torch
import sys
import shutil
import torch.nn.functional as F
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from PIL import Image
from diffusers import DiffusionPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps,
    utils
)

SPECIAL = {"<|startoftext|>", "<|endoftext|>"}


def safe_token(token):
    token = token.replace("</w>", "").replace("/w", "")
    return re.sub(r'[<>:"/\\|?*]', '', token)


def apply_heatmap(tensor):
    arr = tensor.cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() -
                               arr.min() + 1e-8)
    cmap = plt.get_cmap('inferno')
    colored = cmap(arr)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return colored


def patched_save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()

    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(
        token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(
        token_id) for token_id in token_ids]

    os.makedirs(base_dir, exist_ok=True)

    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(
            2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0

    for timestep, layers in attn_maps.items():
        for layer, attn_map in layers.items():
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]

            resized_attn_map = F.interpolate(
                attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

    total_attn_map /= total_attn_map_number

    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        utils.save_attention_image(attn_map, tokens, base_dir, to_pil)

    for batch in range(total_attn_map.shape[0]):
        tokens = total_tokens[batch]
        keep_idx = [i for i, t in enumerate(tokens) if t not in SPECIAL]
        attn_slice = total_attn_map[batch,
                                    keep_idx] if keep_idx else total_attn_map[batch]

        overall_map = attn_slice.mean(dim=0)

        overall_map_up = F.interpolate(
            overall_map.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        heatmap_rgb = apply_heatmap(overall_map_up)

        to_pil(heatmap_rgb).save(os.path.join(
            base_dir, "overall-heatmap.png"))


def patched_save_attention_image(a, tokens, batch_dir, to_pil):
    for i, token in enumerate(tokens):
        token = safe_token(token)

        attn_resized = F.interpolate(
            a[i].unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)

        heatmap_rgb = apply_heatmap(attn_resized)

        to_pil(heatmap_rgb).save(
            os.path.join(batch_dir, f'token-{i}-{token}.png')
        )


def run(prompt="Image of cat sitting on a window sill", seed=482342374238978974, path="./data"):
    if os.path.exists(path):
        shutil.rmtree(path)
    utils.save_attention_maps = patched_save_attention_maps
    utils.save_attention_image = patched_save_attention_image

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    pipe = init_pipeline(pipe)

    prompts = [
        prompt,
    ]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = pipe(
        prompts,
        num_inference_steps=15,
        generator=generator
    ).images

    filtered_maps = attn_maps
    if attn_maps:
        last_key = list(attn_maps.keys())[-1]
        filtered_maps = {last_key: attn_maps[last_key]}
    utils.save_attention_maps(filtered_maps, pipe.tokenizer,
                              prompts, base_dir=f"{path}/heatmaps", unconditional=True)

    for batch, image in enumerate(images):
        image.save(f'{path}/final.png')


if __name__ == "__main__":
    prompt = sys.argv[1]
    seed = sys.argv[2]
    path = sys.argv[3]
    run(prompt, int(seed), path)
