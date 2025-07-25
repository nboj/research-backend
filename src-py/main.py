import sys
from zennit.layer import Sum
from torchvision.models.mobilenetv2 import InvertedResidual
from zennit.canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from torchvision.transforms import ToPILImage
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from diffusers import StableDiffusionPipeline


def save_heatmap(R, filepath, dpi=150):
    """
    Save an LRP relevance map as a red/blue PNG.

    Parameters
    ----------
    R : np.ndarray or torch.Tensor  # shape (H, W) or (C, H, W)
        Relevance scores.
    filepath : str
        Output filename, e.g. "lrp_heatmap.png".
    dpi : int, optional
        Resolution used to convert figure size → pixel size.
    """
    if hasattr(R, "detach"):
        R = R.detach().cpu().numpy()
    if R.ndim == 3:
        R = R.sum(axis=0)
    R = numpy.asarray(R)

    b = 10 * (numpy.abs(R)**3).mean() ** (1/3)

    cmap_arr = plt.cm.seismic(numpy.linspace(0, 1, plt.cm.seismic.N))
    cmap_arr[:, :3] *= 0.85
    cmap = ListedColormap(cmap_arr)
    print(R.shape)

    h_px, w_px = R.shape
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(R, cmap=cmap, vmin=-b, vmax=b, interpolation="nearest")

    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


class MobileNetResidualCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._map)

    @staticmethod
    def _map(name, module):
        if isinstance(module, InvertedResidual) and module.use_res_connect:
            return {
                "forward": MobileNetResidualCanonizer.forward.__get__(module),
                "canonizer_sum": Sum(),
            }
        return None

    @staticmethod
    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = self.canonizer_sum(torch.stack([x, out], dim=-1))
        return out


def generate_sd_image(prompt: str, seed: int, steps: int = 30,
                      out_path: str = "sd_image.png") -> Image.Image:
    """Generate an image with SD‑v1‑4 and save it."""
    SD_MODEL_ID = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch_dtype,
        safety_checker=None,        # <- drop NSFW checker for speed (optional)
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # lower VRAM
    # pipe.enable_model_cpu_offload()  # (needs accelerate >= 0.20)

    g = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, num_inference_steps=steps,
                 generator=g).images[0]          # PIL.Image

    image.save(out_path)
    return image


def run(prompt="Image of cat sitting on a window sill", seed=482342374238978974, path="/data"):
    matplotlib.use("Agg")
    image = generate_sd_image(prompt, seed, out_path=f"{path}/image.png")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights).eval().to("cpu")

    x = weights.transforms()(image).unsqueeze(0).cpu()

    # canonizer = SequentialMergeBatchNorm()
    # composite = EpsilonPlus(
    #    canonizers=[canonizer])

    canonizer = CompositeCanonizer((
        SequentialMergeBatchNorm(),
        MobileNetResidualCanonizer(),
    ))
    composite = EpsilonPlus(canonizers=[canonizer])

    with Gradient(model, composite) as attr:
        relevance = attr(x)[1]

    save_heatmap(relevance[0], f"{path}/heatmap.png")


if __name__ == "__main__":
    prompt = sys.argv[1]
    seed = sys.argv[2]
    path = sys.argv[3]
    run(prompt, int(seed), path)
