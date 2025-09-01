import torch
import sys
import shutil
import torch.nn.functional as F
import os
from daam import trace, set_seed
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
import torch
from daam import GenerationExperiment, trace
from pathlib import Path
import json
from collections import defaultdict

from daam.utils import compute_token_merge_indices


def write_token_map(path: str, prompt: str, tokenizer):
    """
    Writes ./data/<id>/token_map.json with:
      {
        "prompt": "...",
        "tokens_all": ["ĠA","Ġdog","Ġruns","Ġacross","Ġthe","Ġfield"],
        "mapping": [
          {"word": "A", "word_idx": 0, "token_indices": [0], "tokens": ["ĠA"], "heatmap": "a.heat_map.png"},
          {"word": "dog", "word_idx": 0, "token_indices": [1], "tokens": ["Ġdog"], "heatmap": "dog.heat_map.png"},
          ...
        ]
      }
    """
    out_dir = Path(path)
    toks = tokenizer.tokenize(prompt)  # CLIP display tokens (no specials)

    # keep track of repeated words: word_idx = 0,1,2,...
    counts = defaultdict(int)
    mapping = []
    for word in prompt.split():
        idx = counts[word]
        counts[word] += 1
        try:
            merge = compute_token_merge_indices(
                tokenizer, prompt, word, word_idx=idx)
            entry = {
                "word": word,
                "word_idx": idx,
                "token_indices": merge,
                "tokens": [toks[i] for i in merge],
                "heatmap": f"{word.lower()}.heat_map.png",
            }
        except Exception as e:
            # still emit a record so Rust can skip/match gracefully
            entry = {
                "word": word,
                "word_idx": idx,
                "token_indices": [],
                "tokens": [],
                "heatmap": f"{word.lower()}.heat_map.png",
                "error": str(e),
            }
        mapping.append(entry)

    with (out_dir / "token_map.json").open("w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "tokens_all": toks,
                  "mapping": mapping}, f, ensure_ascii=False, indent=2)


def run(prompt="Image of cat sitting on a window sill", seed=482342374238978974, path="./data"):
    if os.path.exists(path):
        shutil.rmtree(path)

    model_id = 'stabilityai/stable-diffusion-2-1'
    device = 'cuda'

    pipe = DiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True, torch_dtype=torch.float16, use_safetensors=True, variant='fp16')
    pipe = pipe.to(device)

    gen = set_seed(seed)  # for reproducibility
    with trace(pipe) as tc:
        pipe(prompt)
        exp = tc.to_experiment(path)
        exp.save()  # experiment-dir now contains all the data and heat maps
        write_token_map(path, prompt, pipe.tokenizer)

    # exp = GenerationExperiment.load(path)  # load the experiment
    # print(exp)


if __name__ == "__main__":
    prompt = sys.argv[1]
    seed = sys.argv[2]
    path = sys.argv[3]
    run(prompt, int(seed), path)
